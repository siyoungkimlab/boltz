"""boltz screen: predict a complex with each ligand of a list.

The input YAML is a template: its proteins, and one ligand entry whose SMILES
is swapped for each ligand of the list. Everything else in the template is kept
for every ligand, such as a pocket constraint on the ligand or an affinity
property. All the predictions share one MSA, computed once, and one ``boltz
predict`` run, so the model is loaded once.
"""

import copy
import csv
import re
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import click
import yaml
from rdkit import Chem, RDLogger

from boltz.batch import (
    CONFIDENCE_SCORES,
    DEFAULT_MAX_DISTANCE,
    DISTANCE_FIELDS,
    SMILES_FIELDS,
    affinity_binders,
    chain_ids,
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

HELP = """Predict a complex with each ligand of a list.

DATA is a YAML file with the proteins and one ligand entry, whose SMILES is
replaced by each ligand of --ligands. Everything else in it applies to every
ligand, such as a pocket constraint or an affinity property. The MSA is
computed once and the model loaded once. Takes every option of boltz predict.
"""


class Ligand(NamedTuple):
    """A ligand of the list, by the name its prediction folder gets."""

    name: str
    smiles: str


def read_ligands(path: Path) -> list[tuple[str, str]]:
    """Read (SMILES, name) pairs from a ligand list; the name may be empty.

    A ``.csv`` file has a ``smiles`` column and optionally a ``name`` column.
    Any other file has one ligand per line: a SMILES, then optionally its name.
    Blank lines and lines starting with ``#`` are skipped.
    """
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            columns = {c.strip().lower(): c for c in reader.fieldnames or []}
            if "smiles" not in columns:
                msg = f"{path.name} has no smiles column."
                raise click.UsageError(msg)
            name_column = columns.get("name") or columns.get("id")
            return [
                (
                    (row[columns["smiles"]] or "").strip(),
                    (row[name_column] or "").strip() if name_column else "",
                )
                for row in reader
                if (row[columns["smiles"]] or "").strip()
            ]

    ligands = []
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        # The SMILES ends at the first space or tab; the rest is the name.
        smiles, *rest = text.split(maxsplit=1)
        if not ligands and smiles.lower() == "smiles":
            continue  # a header line
        ligands.append((smiles, rest[0].strip() if rest else ""))
    return ligands


def name_ligands(entries: list[tuple[str, str]]) -> list[Ligand]:
    """Name each ligand for its folder: safe for filenames, and unique.

    A ligand without a name is ``lig_001``, ``lig_002``, and so on.
    """
    width = max(3, len(str(len(entries))))
    used = set()
    ligands = []
    for index, (smiles, name) in enumerate(entries, start=1):
        base = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or (
            f"lig_{index:0{width}d}"
        )
        unique, suffix = base, 2
        while unique in used:
            unique, suffix = f"{base}_{suffix}", suffix + 1
        used.add(unique)
        ligands.append(Ligand(unique, smiles))
    return ligands


def split_valid(ligands: list[Ligand]) -> tuple[list[Ligand], list[Ligand]]:
    """Separate the ligands RDKit can read from those it cannot."""
    RDLogger.DisableLog("rdApp.*")
    try:
        valid = [lig for lig in ligands if Chem.MolFromSmiles(lig.smiles) is not None]
    finally:
        RDLogger.EnableLog("rdApp.*")
    invalid = [lig for lig in ligands if lig not in valid]
    return valid, invalid


def find_ligand_entry(schema: dict, ligand_id: Optional[str]) -> tuple[int, str]:
    """Find the template's ligand entry to swap, as its index and chain id.

    Without ``ligand_id``, it is the template's only ligand entry.
    """
    entries = [
        (index, item["ligand"])
        for index, item in enumerate(schema["sequences"])
        if "ligand" in item
    ]
    if ligand_id is None:
        if len(entries) != 1:
            found = (
                "no ligand entry"
                if not entries
                else "ligands " + ", ".join(chain_ids(e)[0] for _, e in entries)
            )
            msg = (
                f"The template has {found}: it needs one ligand entry to swap each "
                "ligand into, chosen with --ligand when there are several."
            )
            raise click.UsageError(msg)
        index, entry = entries[0]
        return index, chain_ids(entry)[0]
    for index, entry in entries:
        if ligand_id in chain_ids(entry):
            return index, ligand_id
    msg = f"--ligand {ligand_id} is not a ligand chain of the template."
    raise click.UsageError(msg)


def check_constraints(schema: dict, ligand_id: str, model: str) -> Optional[dict]:
    """Fail early on what would fail for every ligand, not once per ligand.

    Returns the pocket constraint on the ligand, if the template has one with
    contacts, for the summary to measure.
    """
    pockets = [c["pocket"] for c in schema.get("constraints") or [] if "pocket" in c]
    if any("contacts" not in pocket for pocket in pockets):
        msg = (
            "Every pocket constraint needs contacts in a screen; to fill them in "
            "residue by residue, use boltz pointprobe."
        )
        raise click.UsageError(msg)
    if model == "boltz1":
        if schema.get("properties"):
            msg = "Affinity is only predicted by Boltz-2; use --model boltz2."
            raise click.UsageError(msg)
        if len(pockets) > 1 or any(
            float(p.get("max_distance", DEFAULT_MAX_DISTANCE)) != DEFAULT_MAX_DISTANCE
            for p in pockets
        ):
            msg = "Boltz-1 supports one pocket constraint, at max_distance 6."
            raise click.UsageError(msg)
    return next((p for p in pockets if str(p["binder"]) == ligand_id), None)


def check_screen_bonds(schema: dict, ligand_id: str) -> None:
    """Refuse a bond to the screened ligand, then check the other bonds.

    Each ligand names its atoms differently, so one bond constraint cannot
    name the same atom in all of them.
    """
    for constraint in schema.get("constraints") or []:
        bond = constraint.get("bond") or {}
        chains = {str((bond.get(key) or [None])[0]) for key in ("atom1", "atom2")}
        if ligand_id in chains:
            msg = (
                f"A bond to the screened ligand {ligand_id} cannot be screened: "
                "each ligand names its atoms differently, so one bond constraint "
                "cannot name the same atom in all of them."
            )
            raise click.UsageError(msg)
    check_bonds(schema)


def check_renamed(inputs_dir: Path, ligands: list[Ligand], index: int) -> None:
    """Refuse a ligand name whose SMILES changed since the last run here.

    Boltz reuses an input it has processed before, by name, so a changed SMILES
    under the same name would be predicted as the old one.
    """
    changed = []
    for ligand in ligands:
        old = inputs_dir / f"{ligand.name}.yaml"
        if old.exists():
            entry = yaml.safe_load(old.read_text())["sequences"][index]["ligand"]
            if entry.get("smiles") != ligand.smiles:
                changed.append(ligand.name)
    if changed:
        msg = (
            f"{', '.join(changed)} had other SMILES in the last run into this "
            "--out_dir, whose processed inputs Boltz would reuse; rename them or "
            "use a new --out_dir."
        )
        raise click.UsageError(msg)


def write_summary(
    results: Path,
    ligands: list[Ligand],
    ligand_id: str,
    pocket: Optional[dict],
    output_format: str,
    *,
    affinity_binder: bool = False,
) -> Path:
    """Tabulate every structure: its ligand, its SMILES, pocket distances and scores."""
    contacts = (
        [(str(c), int(r)) for c, r in pocket["contacts"] if isinstance(r, int)]
        if pocket
        else []
    )
    max_distance = float((pocket or {}).get("max_distance", DEFAULT_MAX_DISTANCE))

    rows = []
    for ligand in ligands:
        folder = results / "predictions" / ligand.name
        affinity = read_affinity(folder, ligand.name)
        for model, structure, confidence in predicted_models(
            folder, ligand.name, output_format
        ):
            row = {
                "ligand": ligand.name,
                "model": model,
                "structure": structure.name if structure else "",
            }
            row.update(
                ligand_smiles_columns(
                    structure, ligand_id, ligand.smiles, affinity_binder
                )
            )
            if contacts:
                row.update(
                    distance_columns(structure, ligand_id, contacts, max_distance)
                )
            row.update(score_columns(confidence, affinity))
            rows.append(row)

    fields = ["ligand", "model", "structure", *SMILES_FIELDS]
    if contacts:
        fields += DISTANCE_FIELDS
    fields += CONFIDENCE_SCORES
    return write_csv(results / "screen_summary.csv", rows, fields)


def make_screen_command(predict: click.Command, compute_msa: Callable) -> click.Command:
    """Build ``boltz screen`` from ``boltz predict``, sharing its options."""
    ligands_option = click.Option(
        ["--ligands"],
        type=click.Path(exists=True, dir_okay=False),
        required=True,
        help=(
            "The ligands to screen: a .smi or .txt file with a SMILES and "
            "optionally a name on each line, or a .csv file with a smiles and "
            "optionally a name column."
        ),
    )
    ligand_option = click.Option(
        ["--ligand"],
        type=str,
        default=None,
        help=(
            "The template's ligand chain to swap each ligand into. Default is "
            "the template's only ligand."
        ),
    )

    def screen(**options: object) -> None:
        ligands_path = Path(str(options.pop("ligands"))).expanduser()
        ligand_choice = options.pop("ligand")
        data = Path(str(options["data"])).expanduser()
        if data.suffix.lower() not in (".yml", ".yaml"):
            msg = "screen takes a YAML template, with a ligand entry to swap."
            raise click.UsageError(msg)

        schema = yaml.safe_load(data.read_text())
        index, ligand_id = find_ligand_entry(schema, ligand_choice)
        pocket = check_constraints(schema, ligand_id, str(options["model"]))
        check_screen_bonds(schema, ligand_id)

        ligands, invalid = split_valid(name_ligands(read_ligands(ligands_path)))
        for ligand in invalid:
            click.echo(f"Skipping {ligand.name}: RDKit cannot read {ligand.smiles}")
        if not ligands:
            msg = f"No ligand of {ligands_path.name} has a SMILES RDKit can read."
            raise click.UsageError(msg)

        name = data.stem
        results, inputs_dir = results_paths(options, f"{name}_screen", "screen_inputs")
        check_renamed(inputs_dir, ligands, index)

        share_msa(schema, name, results / "msa", options, compute_msa)
        inputs = {}
        for ligand in ligands:
            variant = copy.deepcopy(schema)
            entry = variant["sequences"][index]["ligand"]
            entry.pop("ccd", None)
            entry["smiles"] = ligand.smiles
            inputs[ligand.name] = variant
        write_inputs(inputs, inputs_dir)
        click.echo(
            f"Screen: predicting {len(ligands)} ligands as {ligand_id}, "
            f"{options['diffusion_samples']} structure(s) each"
            + (f", skipping {len(invalid)} RDKit cannot read." if invalid else ".")
        )

        run_predict(predict, options, inputs_dir)

        summary = write_summary(
            results,
            ligands,
            ligand_id,
            pocket,
            str(options["output_format"]),
            affinity_binder=ligand_id in affinity_binders(schema),
        )
        click.echo(f"Screen summary written to {summary}.")

    return click.Command(
        name="screen",
        callback=screen,
        params=[*predict.params, ligands_option, ligand_option],
        help=HELP,
        short_help="Predict a complex with each ligand of a list.",
    )
