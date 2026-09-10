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
import csv
import hashlib
import json
import re
import shlex
import sqlite3
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import click
import gemmi
import numpy as np
import yaml
from rdkit import Chem

from boltz.data import const

CONFIDENCE_SCORES = (
    "confidence_score",
    "ptm",
    "iptm",
    "ligand_iptm",
    "protein_iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
)
AFFINITY_SCORES = ("affinity_pred_value", "affinity_probability_binary")
DISTANCES = ("min_distance", "max_contact_distance", "com_distance", "ca_com_distance")
STRUCTURE_SUFFIXES = {"mae": ".mae", "dms": ".dms", "pdb": ".pdb", "mmcif": ".cif"}
DEFAULT_MAX_DISTANCE = 6.0

HELP = """Probe every protein residue as a pocket for a ligand.

Runs one prediction per residue, with a pocket constraint between the ligand
and that residue (and with --window, the residues after it), sharing one MSA
and one loaded model. DATA is a YAML file. A pocket constraint in it without
contacts names the ligand to probe with as its binder, and its other fields
(max_distance, force) apply to every residue. Without one, the input's only
ligand, or --binder, is probed with the default pocket. Takes every option of
boltz predict, and --probe to choose residues.
"""


class ProbeAtom(NamedTuple):
    """An atom of a predicted structure, as far as the distances need it."""

    chain: str
    resid: int
    name: str
    mass: float
    xyz: np.ndarray


class Probe(NamedTuple):
    """One prediction: the residue probed and the contacts constrained for it."""

    chain: str
    resid: int
    contacts: list[int]


def _chain_ids(entry: dict) -> list[str]:
    """Return the chain ids of a sequence entry, which may name several."""
    ids = entry["id"]
    return [str(i) for i in ids] if isinstance(ids, list) else [str(ids)]


def _chains_of_kind(schema: dict, kind: str) -> dict[str, dict]:
    """Map each chain id of one kind, such as protein, to its YAML entry."""
    chains = {}
    for item in schema["sequences"]:
        for item_kind, entry in item.items():
            if item_kind == kind:
                for chain in _chain_ids(entry):
                    chains[chain] = entry
    return chains


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


def add_template(schema: dict, binder: Optional[str]) -> tuple[int, dict]:
    """Add a default pocket constraint without contacts, for the binder.

    Without ``binder``, it is the input's only ligand.
    """
    if binder is None:
        ligands = list(_chains_of_kind(schema, "ligand"))
        if len(ligands) != 1:
            found = f"ligands {', '.join(ligands)}" if ligands else "no ligand"
            msg = (
                f"The input has {found}: name the chain to probe with using "
                "--binder, or with a pocket constraint without contacts."
            )
            raise click.UsageError(msg)
        binder = ligands[0]
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


def check_boltz1(schema: dict, template: dict) -> None:
    """Fail early on what Boltz-1 does not support, not once per residue."""
    others = [
        c
        for c in schema.get("constraints") or []
        if "pocket" in c and "contacts" in c["pocket"]
    ]
    if others:
        msg = (
            "Boltz-1 supports one pocket constraint, and pointprobe adds it; "
            "remove the other pocket constraints or use --model boltz2."
        )
        raise click.UsageError(msg)
    if (
        float(template.get("max_distance", DEFAULT_MAX_DISTANCE))
        != DEFAULT_MAX_DISTANCE
    ):
        msg = "Boltz-1 supports only max_distance 6; use --model boltz2."
        raise click.UsageError(msg)


def share_msa(
    schema: dict, name: str, msa_dir: Path, options: dict, compute_msa: Callable
) -> None:
    """Compute the MSA once and point every protein without one at it.

    The MSA is computed as ``boltz predict`` would for the input, pairing all
    its proteins without an MSA, and saved under a hash of their sequences so
    a rerun reuses it and an edited input does not.
    """
    missing = [
        entry
        for item in schema["sequences"]
        for kind, entry in item.items()
        if kind == "protein" and "msa" not in entry
    ]
    # Without the server, boltz predict reports the missing MSAs itself.
    if not missing or not options["use_msa_server"]:
        return

    sequences = list(dict.fromkeys(entry["sequence"] for entry in missing))
    digest = hashlib.sha1("|".join(sequences).encode()).hexdigest()[:10]  # noqa: S324
    paths = {
        sequence: (msa_dir / f"{name}_{digest}_{index}.csv").resolve()
        for index, sequence in enumerate(sequences)
    }
    if all(path.exists() for path in paths.values()):
        click.echo(f"Reusing the MSA computed for {name}.")
    else:
        if options["msa_server_username"] and options["api_key_value"]:
            msg = "Use either basic authentication or an API key for the MSA server."
            raise click.UsageError(msg)
        msa_dir.mkdir(parents=True, exist_ok=True)
        compute_msa(
            data={path.stem: sequence for sequence, path in paths.items()},
            target_id=name,
            msa_dir=msa_dir,
            msa_server_url=options["msa_server_url"],
            msa_pairing_strategy=options["msa_pairing_strategy"],
            msa_server_username=options["msa_server_username"],
            msa_server_password=options["msa_server_password"],
            api_key_header=options["api_key_header"],
            api_key_value=options["api_key_value"],
        )
    for entry in missing:
        entry["msa"] = str(paths[entry["sequence"]])


def write_inputs(
    schema: dict,
    template_index: int,
    probes: list[Probe],
    name: str,
    inputs_dir: Path,
) -> dict[str, Probe]:
    """Write one input per probe, with its contacts filled in.

    Returns each probe by its record id, which names its prediction folder.
    """
    inputs_dir.mkdir(parents=True, exist_ok=True)
    # A rerun may probe other residues, so start from an empty folder.
    for old in inputs_dir.glob("*.yaml"):
        old.unlink()

    records = {}
    for probe in probes:
        variant = copy.deepcopy(schema)
        variant["constraints"][template_index]["pocket"]["contacts"] = [
            [probe.chain, resid] for resid in probe.contacts
        ]
        record_id = f"{name}_{probe.chain}_{probe.resid}"
        with (inputs_dir / f"{record_id}.yaml").open("w") as f:
            yaml.safe_dump(variant, f, sort_keys=False)
        records[record_id] = probe
    return records


def read_atoms(path: Path) -> list[ProbeAtom]:
    """Read every atom of a predicted structure, in any output format."""
    if path.suffix == ".dms":
        con = sqlite3.connect(path)
        try:
            rows = con.execute(
                "select chain, resid, name, mass, x, y, z from particle"
            ).fetchall()
        finally:
            con.close()
        return [
            ProbeAtom(str(c), int(r), str(n).strip(), float(m), np.array(xyz))
            for c, r, n, m, *xyz in rows
        ]

    if path.suffix == ".mae":
        periodic_table = Chem.GetPeriodicTable()
        block = path.read_text().split(" m_atom[")[1].split(":::\n")
        columns = [c.strip() for c in block[0].splitlines()[1:] if c.strip()]
        atoms = []
        for line in block[1].splitlines():
            if not line.strip():
                continue
            row = dict(zip(["index", *columns], shlex.split(line)))
            atomic_number = int(row["i_m_atomic_number"])
            atoms.append(
                ProbeAtom(
                    row["s_m_chain_name"],
                    int(row["i_m_residue_number"]),
                    row["s_m_pdb_atom_name"].strip(),
                    periodic_table.GetAtomicWeight(atomic_number)
                    if atomic_number
                    else 0.0,
                    np.array([float(row[f"r_m_{axis}_coord"]) for axis in "xyz"]),
                )
            )
        return atoms

    structure = gemmi.read_structure(str(path))
    return [
        ProbeAtom(
            chain.name,
            residue.seqid.num,
            atom.name,
            atom.element.weight,
            np.array(atom.pos.tolist()),
        )
        for chain in structure[0]
        for residue in chain
        for atom in residue
    ]


def _center_of_mass(atoms: list[ProbeAtom]) -> np.ndarray:
    xyz = np.array([atom.xyz for atom in atoms])
    masses = np.array([atom.mass for atom in atoms])
    return np.average(xyz, axis=0, weights=masses if masses.sum() > 0 else None)


def probe_distances(
    path: Path, binder: str, chain: str, contacts: list[int]
) -> Optional[dict[str, Optional[float]]]:
    """Measure how close the binder came to the contact residues.

    ``min_distance`` is the shortest distance between any atom of the binder
    and of the contacts. ``max_contact_distance`` is the binder's closest
    approach to each contact, at the contact it came least close to: the
    distance a pocket constraint limits, since it holds for every contact.
    ``com_distance`` is between the centers of mass of the binder and of the
    contacts, and ``ca_com_distance`` from the contacts' mean CA to the
    binder's center of mass.
    """
    atoms = read_atoms(path)
    ligand = [atom for atom in atoms if atom.chain == binder]
    residues = [
        [atom for atom in atoms if atom.chain == chain and atom.resid == resid]
        for resid in contacts
    ]
    if not ligand or not all(residues):
        return None

    ligand_xyz = np.array([atom.xyz for atom in ligand])
    ligand_com = _center_of_mass(ligand)
    closest = [
        np.linalg.norm(
            ligand_xyz[:, None, :] - np.array([a.xyz for a in residue])[None], axis=-1
        ).min()
        for residue in residues
    ]
    contact_atoms = [atom for residue in residues for atom in residue]
    cas = [atom.xyz for residue in residues for atom in residue if atom.name == "CA"]
    return {
        "min_distance": float(min(closest)),
        "max_contact_distance": float(max(closest)),
        "com_distance": float(
            np.linalg.norm(ligand_com - _center_of_mass(contact_atoms))
        ),
        "ca_com_distance": float(np.linalg.norm(np.mean(cas, axis=0) - ligand_com))
        if len(cas) == len(residues)
        else None,
    }


def _format_contacts(contacts: list[int]) -> str:
    return str(contacts[0]) if len(contacts) == 1 else f"{contacts[0]}-{contacts[-1]}"


def write_summary(
    results: Path,
    records: dict[str, Probe],
    chains: dict[str, dict],
    template: dict,
    output_format: str,
) -> Path:
    """Tabulate every structure: its probed residue, distances and scores."""
    binder = str(template["binder"])
    max_distance = float(template.get("max_distance", DEFAULT_MAX_DISTANCE))
    suffix = STRUCTURE_SUFFIXES.get(output_format)

    rows = []
    for record_id, probe in records.items():
        folder = results / "predictions" / record_id
        affinity_path = folder / f"affinity_{record_id}.json"
        affinity = (
            json.loads(affinity_path.read_text()) if affinity_path.exists() else {}
        )
        confidences = sorted(
            folder.glob(f"confidence_{record_id}_model_*.json"),
            key=lambda p: int(p.stem.rsplit("_", 1)[1]),
        )
        for confidence_path in confidences:
            model = int(confidence_path.stem.rsplit("_", 1)[1])
            confidence = json.loads(confidence_path.read_text())
            structure = folder / f"{record_id}_model_{model}{suffix}"
            distances = (
                probe_distances(structure, binder, probe.chain, probe.contacts)
                if suffix and structure.exists()
                else None
            ) or {}
            row = {
                "probe_chain": probe.chain,
                "probe_residue": probe.resid,
                "probe_residue_name": residue_name(chains[probe.chain], probe.resid),
                "contacts": _format_contacts(probe.contacts),
                "model": model,
                "structure": structure.name if structure.exists() else "",
            }
            for key in DISTANCES:
                value = distances.get(key)
                row[key] = "" if value is None else round(value, 3)
            row["within_max_distance"] = (
                ""
                if row["max_contact_distance"] == ""
                else row["max_contact_distance"] <= max_distance
            )
            row.update({key: confidence.get(key, "") for key in CONFIDENCE_SCORES})
            row.update({key: affinity.get(key, "") for key in AFFINITY_SCORES})
            rows.append(row)

    path = results / "pointprobe_summary.csv"
    fields = [
        "probe_chain",
        "probe_residue",
        "probe_residue_name",
        "contacts",
        "model",
        "structure",
        *DISTANCES,
        "within_max_distance",
        *CONFIDENCE_SCORES,
    ]
    if any(row[key] != "" for row in rows for key in AFFINITY_SCORES):
        fields += list(AFFINITY_SCORES)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def make_pointprobe_command(
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

    def pointprobe(**options: object) -> None:
        spec = options.pop("probe")
        binder_chain = options.pop("binder")
        window = int(options.pop("window"))
        data = Path(str(options["data"])).expanduser()
        if data.suffix.lower() not in (".yml", ".yaml"):
            msg = "pointprobe takes a YAML file, which can hold pocket constraints."
            raise click.UsageError(msg)

        schema = yaml.safe_load(data.read_text())
        found = find_template(schema)
        if found is None:
            template_index, template = add_template(schema, binder_chain)
        else:
            template_index, template = found
            if binder_chain is not None and binder_chain != str(template["binder"]):
                msg = (
                    f"--binder {binder_chain} differs from the binder of the "
                    f"input's pocket constraint, {template['binder']}."
                )
                raise click.UsageError(msg)
        binder = str(template["binder"])
        all_chains = {
            chain
            for item in schema["sequences"]
            for entry in item.values()
            for chain in _chain_ids(entry)
        }
        if binder not in all_chains:
            msg = f"The binder {binder} is not a chain of the input."
            raise click.UsageError(msg)
        chains = _chains_of_kind(schema, "protein")
        targets = parse_probe(spec, chains, binder)
        if not targets:
            msg = "There are no protein residues to probe."
            raise click.UsageError(msg)
        if options["model"] == "boltz1":
            check_boltz1(schema, template)
        probes = make_probes(targets, chains, window)

        # boltz predict names its results after the folder of inputs. Each
        # window gets its own, since predictions are named by probed residue.
        name = data.stem
        run = f"{name}_pointprobe" + (f"_w{window}" if window > 1 else "")
        results = Path(str(options["out_dir"])).expanduser() / f"boltz_results_{run}"
        inputs_dir = results / "pointprobe_inputs" / run

        share_msa(schema, name, results / "msa", options, compute_msa)
        records = write_inputs(schema, template_index, probes, name, inputs_dir)
        click.echo(
            f"Pointprobe: probing {len(records)} residues with {binder}, "
            f"{window} residue(s) per pocket, "
            f"{options['diffusion_samples']} structure(s) each."
        )

        options["data"] = str(inputs_dir)
        click.get_current_context().invoke(predict, **options)

        summary = write_summary(
            results, records, chains, template, str(options["output_format"])
        )
        click.echo(f"Pointprobe summary written to {summary}.")

    return click.Command(
        name="pointprobe",
        callback=pointprobe,
        params=[*predict.params, probe_option, binder_option, window_option],
        help=HELP,
        short_help="Probe every protein residue as a pocket for a ligand.",
    )
