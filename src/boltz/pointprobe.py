"""boltz pointprobe: probe every residue as a pocket for a ligand.

Runs one prediction per protein residue, each with a pocket constraint between
the ligand of interest and that residue, or with ``--window`` the residue and
the ones after it. The input YAML may name the ligand with a pocket constraint
that has no ``contacts``: its ``binder`` is the ligand, and its other fields
(``max_distance``, ``force``) are kept for every residue, whose contacts are
filled in. Without one, the input's only ligand (or ``--binder``) is probed with
Boltz's default pocket. All the predictions share one MSA, computed once, and
one ``boltz predict`` run, so the model is loaded once.
"""

import copy
import re
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import click
import yaml

from boltz.batch import (
    CONFIDENCE_SCORES,
    DEFAULT_MAX_DISTANCE,
    DISTANCE_FIELDS,
    SMILES_FIELDS,
    affinity_binders,
    all_chain_ids,
    chains_of_kind,
    check_bonds,
    distance_columns,
    ligand_smiles_columns,
    predicted_models,
    read_affinity,
    results_paths,
    run_predict,
    score_columns,
    share_msa,
    write_csv,
    write_inputs,
)
from boltz.data import const

HELP = """Probe every protein residue as a pocket for a ligand.

Runs one prediction per residue, with a pocket constraint between the ligand
and that residue (and with --window, the residues after it), sharing one MSA
and one loaded model. DATA is a YAML file. A pocket constraint in it without
contacts names the ligand to probe with as its binder, and its other fields
(max_distance, force) apply to every residue. Without one, the input's only
ligand, or --binder, is probed with the default pocket. Takes every option of
boltz predict, and --probe to choose residues.
"""


class Probe(NamedTuple):
    """One prediction: the residue probed and the contacts constrained for it."""

    chain: str
    resid: int
    contacts: list[int]


def find_template(schema: dict) -> Optional[tuple[int, dict]]:
    """Find the pocket constraint without contacts, if the input has one."""
    templates = [
        (index, constraint["pocket"])
        for index, constraint in enumerate(schema.get("constraints") or [])
        if "pocket" in constraint and "contacts" not in constraint["pocket"]
    ]
    if len(templates) > 1:
        msg = (
            f"Found {len(templates)} pocket constraints without contacts; at most "
            "one names the ligand to probe with."
        )
        raise click.UsageError(msg)
    return templates[0] if templates else None


def choose_binder(schema: dict, binder: Optional[str]) -> str:
    """Return the chain to probe with: ``binder``, or the input's only ligand."""
    if binder is not None:
        return binder
    ligands = list(chains_of_kind(schema, "ligand"))
    if len(ligands) != 1:
        found = f"ligands {', '.join(ligands)}" if ligands else "no ligand"
        msg = (
            f"The input has {found}: name the chain to probe with using "
            "--binder, or with a pocket constraint without contacts."
        )
        raise click.UsageError(msg)
    return ligands[0]


def add_template(schema: dict, binder: str) -> tuple[int, dict]:
    """Add a default pocket constraint without contacts, for the binder."""
    click.echo(
        f"Probing with {binder} using the default pocket: max_distance "
        f"{DEFAULT_MAX_DISTANCE:g}, no force."
    )
    schema["constraints"] = [
        *(schema.get("constraints") or []),
        {"pocket": {"binder": binder}},
    ]
    index = len(schema["constraints"]) - 1
    return index, schema["constraints"][index]["pocket"]


def residue_name(entry: dict, resid: int) -> str:
    """Name a residue by its CCD code, following any modification."""
    for modification in entry.get("modifications") or []:
        if int(modification["position"]) == resid:
            return str(modification["ccd"])
    letter = entry["sequence"][resid - 1]
    return const.prot_letter_to_token.get(letter, "UNK")


def parse_probe(
    spec: Optional[str], chains: dict[str, dict], binder: str
) -> list[tuple[str, int]]:
    """List the residues to probe, as (chain, residue number) pairs.

    ``spec`` is a comma-separated list of chains and residue ranges, such as
    ``A,B:10-50,B:62``. Without one, every residue of every protein chain but
    the binder is probed.
    """
    if not spec:
        return [
            (chain, resid)
            for chain, entry in chains.items()
            if chain != binder
            for resid in range(1, len(entry["sequence"]) + 1)
        ]

    targets, seen = [], set()
    for token in (t.strip() for t in spec.split(",")):
        chain, _, span = token.partition(":")
        if chain not in chains:
            msg = f"{chain} is not a protein chain; they are {', '.join(chains)}."
            raise click.BadParameter(msg, param_hint="--probe")
        length = len(chains[chain]["sequence"])
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", span) if span else None
        if span and match is None:
            msg = f"{token}: expected CHAIN, CHAIN:RESIDUE or CHAIN:FIRST-LAST."
            raise click.BadParameter(msg, param_hint="--probe")
        first = int(match[1]) if match else 1
        last = int(match[2] or match[1]) if match else length
        if not 1 <= first <= last <= length:
            msg = f"{token}: chain {chain} has residues 1-{length}."
            raise click.BadParameter(msg, param_hint="--probe")
        for resid in range(first, last + 1):
            if (chain, resid) not in seen:
                seen.add((chain, resid))
                targets.append((chain, resid))
    return targets


def make_probes(
    targets: list[tuple[str, int]], chains: dict[str, dict], window: int
) -> list[Probe]:
    """Pair each residue to probe with its window of contacts.

    A window covers the residue and the ``window - 1`` after it in its chain,
    so there is one probe per residue. It never crosses into the next chain:
    near a chain's end it is cut short, down to the last residue alone.
    """
    probes = []
    for chain, resid in targets:
        last = min(resid + window - 1, len(chains[chain]["sequence"]))
        probes.append(Probe(chain, resid, list(range(resid, last + 1))))
    return probes


def check_boltz1(schema: dict, template: dict, probing: bool = True) -> None:
    """Fail early on what Boltz-1 does not support, not once per residue.

    Boltz-1 takes one pocket constraint, at 6 Angstrom. When probing,
    pointprobe adds it; a baseline adds none, so only the input's own count.
    """
    others = [
        c["pocket"]
        for c in schema.get("constraints") or []
        if "pocket" in c and "contacts" in c["pocket"]
    ]
    if probing and others:
        msg = (
            "Boltz-1 supports one pocket constraint, and pointprobe adds it; "
            "remove the other pocket constraints or use --model boltz2."
        )
        raise click.UsageError(msg)
    if len(others) > 1:
        msg = "Boltz-1 supports one pocket constraint; use --model boltz2."
        raise click.UsageError(msg)
    pockets = [template] if probing else others
    if any(
        float(p.get("max_distance", DEFAULT_MAX_DISTANCE)) != DEFAULT_MAX_DISTANCE
        for p in pockets
    ):
        msg = "Boltz-1 supports only max_distance 6; use --model boltz2."
        raise click.UsageError(msg)


def _format_contacts(contacts: list[int]) -> str:
    return str(contacts[0]) if len(contacts) == 1 else f"{contacts[0]}-{contacts[-1]}"


def write_summary(
    results: Path,
    records: dict[str, Probe],
    chains: dict[str, dict],
    template: dict,
    output_format: str,
    *,
    ligand_smiles: Optional[str] = None,
    affinity_binder: bool = False,
) -> Path:
    """Tabulate every structure: its probed residue, SMILES, distances and scores."""
    binder = str(template["binder"])
    max_distance = float(template.get("max_distance", DEFAULT_MAX_DISTANCE))

    rows = []
    for record_id, probe in records.items():
        folder = results / "predictions" / record_id
        affinity = read_affinity(folder, record_id)
        contacts = [(probe.chain, resid) for resid in probe.contacts]
        for model, structure, confidence in predicted_models(
            folder, record_id, output_format
        ):
            row = {
                "probe_chain": probe.chain,
                "probe_residue": probe.resid,
                "probe_residue_name": residue_name(chains[probe.chain], probe.resid),
                "contacts": _format_contacts(probe.contacts),
                "model": model,
                "structure": structure.name if structure else "",
            }
            row.update(
                ligand_smiles_columns(structure, binder, ligand_smiles, affinity_binder)
            )
            row.update(distance_columns(structure, binder, contacts, max_distance))
            row.update(score_columns(confidence, affinity))
            rows.append(row)

    fields = [
        "probe_chain",
        "probe_residue",
        "probe_residue_name",
        "contacts",
        "model",
        "structure",
        *SMILES_FIELDS,
        *DISTANCE_FIELDS,
        *CONFIDENCE_SCORES,
    ]
    return write_csv(results / "pointprobe_summary.csv", rows, fields)


def make_pointprobe_command(  # noqa: C901, PLR0915
    predict: click.Command, compute_msa: Callable
) -> click.Command:
    """Build ``boltz pointprobe`` from ``boltz predict``, sharing its options."""
    probe_option = click.Option(
        ["--probe"],
        type=str,
        default=None,
        help=(
            "The residues to probe, as comma-separated chains and residue ranges, "
            "e.g. A,B:10-50,B:62. Default is every residue of every protein chain."
        ),
    )
    binder_option = click.Option(
        ["--binder"],
        type=str,
        default=None,
        help=(
            "The chain to probe with, when the input has no pocket constraint "
            "without contacts naming it. Default is the input's only ligand."
        ),
    )
    window_option = click.Option(
        ["--window"],
        type=click.IntRange(min=1),
        default=1,
        show_default=True,
        help=(
            "The number of consecutive residues in each pocket: the probed "
            "residue and the ones after it in its chain, fewer at a chain's end."
        ),
    )

    baseline_option = click.Option(
        ["--baseline"],
        is_flag=True,
        help=(
            "Run the same predictions without the probed pocket, one independent "
            "run per residue, to compare with: each residue's row then shows how "
            "close the ligand came to it on its own."
        ),
    )

    def pointprobe(**options: object) -> None:  # noqa: C901, PLR0912, PLR0915
        spec = options.pop("probe")
        binder_chain = options.pop("binder")
        window = int(options.pop("window"))
        baseline = bool(options.pop("baseline"))
        data = Path(str(options["data"])).expanduser()
        if data.suffix.lower() not in (".yml", ".yaml"):
            msg = "pointprobe takes a YAML file, which can hold pocket constraints."
            raise click.UsageError(msg)

        schema = yaml.safe_load(data.read_text())
        found = find_template(schema)
        if found is None:
            binder = choose_binder(schema, binder_chain)
            if baseline:
                template_index, template = None, {"binder": binder}
            else:
                template_index, template = add_template(schema, binder)
        else:
            template_index, template = found
            if binder_chain is not None and binder_chain != str(template["binder"]):
                msg = (
                    f"--binder {binder_chain} differs from the binder of the "
                    f"input's pocket constraint, {template['binder']}."
                )
                raise click.UsageError(msg)
        binder = str(template["binder"])
        if binder not in all_chain_ids(schema):
            msg = f"The binder {binder} is not a chain of the input."
            raise click.UsageError(msg)
        check_bonds(schema)
        chains = chains_of_kind(schema, "protein")
        targets = parse_probe(spec, chains, binder)
        if not targets:
            msg = "There are no protein residues to probe."
            raise click.UsageError(msg)
        if options["model"] == "boltz1":
            check_boltz1(schema, template, probing=not baseline)
        probes = make_probes(targets, chains, window)

        # Each window, and the baseline, gets its own results, since
        # predictions are named by probed residue.
        name = data.stem
        run = (
            f"{name}_pointprobe"
            + ("_baseline" if baseline else "")
            + (f"_w{window}" if window > 1 else "")
        )
        results, inputs_dir = results_paths(options, run, "pointprobe_inputs")

        share_msa(schema, name, results / "msa", options, compute_msa)
        records = {f"{name}_{probe.chain}_{probe.resid}": probe for probe in probes}
        inputs = {}
        for record_id, probe in records.items():
            variant = copy.deepcopy(schema)
            if not baseline:
                variant["constraints"][template_index]["pocket"]["contacts"] = [
                    [probe.chain, resid] for resid in probe.contacts
                ]
            elif template_index is not None:
                # A baseline keeps the input's other constraints, not the probe.
                del variant["constraints"][template_index]
                if not variant["constraints"]:
                    del variant["constraints"]
            inputs[record_id] = variant
        write_inputs(inputs, inputs_dir)
        if baseline:
            click.echo(
                f"Pointprobe baseline: {len(records)} independent runs without the "
                f"probed pocket, one per residue, measuring {binder}'s distance to "
                f"each; {options['diffusion_samples']} structure(s) each."
            )
        else:
            click.echo(
                f"Pointprobe: probing {len(records)} residues with {binder}, "
                f"{window} residue(s) per pocket, "
                f"{options['diffusion_samples']} structure(s) each."
            )

        run_predict(predict, options, inputs_dir)

        binder_entry = chains_of_kind(schema, "ligand").get(binder) or {}
        summary = write_summary(
            results,
            records,
            chains,
            template,
            str(options["output_format"]),
            ligand_smiles=binder_entry.get("smiles"),
            affinity_binder=binder in affinity_binders(schema),
        )
        click.echo(f"Pointprobe summary written to {summary}.")

    return click.Command(
        name="pointprobe",
        callback=pointprobe,
        params=[
            *predict.params,
            probe_option,
            binder_option,
            window_option,
            baseline_option,
        ],
        help=HELP,
        short_help="Probe every protein residue as a pocket for a ligand.",
    )
