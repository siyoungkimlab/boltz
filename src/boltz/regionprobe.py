"""boltz regionprobe: probe every region of a protein as a pocket for a ligand.

Predicts the input without its binder first, splits that structure's residues
into k regions by k-means on their CB atoms (CA for glycine), and runs one
prediction per region, each with a pocket constraint between the binder and
the region's residues nearest its center. Like ``boltz pointprobe``, the
predictions share one MSA and one ``boltz predict`` run; the apo prediction is
a second, earlier run.
"""

import copy
import shlex
import shutil
import sqlite3
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import click
import gemmi
import numpy as np
import yaml
from sklearn.cluster import KMeans

from boltz.batch import (
    BOND_FIELDS,
    CONFIDENCE_SCORES,
    DEFAULT_MAX_DISTANCE,
    DISTANCE_FIELDS,
    SMILES_FIELDS,
    affinity_binders,
    all_chain_ids,
    bond_columns,
    bond_constraints,
    chain_ids,
    chains_of_kind,
    check_bonds,
    check_guidance_weights,
    distance_columns,
    ligand_smiles_columns,
    predicted_models,
    read_affinity,
    read_atoms,
    results_paths,
    run_predict,
    score_columns,
    share_msa,
    write_csv,
    write_inputs,
)
from boltz.pointprobe import (
    add_template,
    check_boltz1,
    choose_binder,
    find_template,
    residue_name,
)

HELP = """Probe every region of a protein as a pocket for a ligand.

Predicts the input without its binder first, splits that structure's residues
into k regions by k-means on their CB atoms (CA for glycine), across all its
protein chains, and runs one prediction per region, with a pocket constraint
between the ligand and the region's residues nearest its center. The
predictions share one MSA and one loaded model. DATA is a YAML file. A pocket
constraint in it without contacts names the ligand to probe with as its
binder, and its other fields (max_distance, force) apply to every region.
Without one, the input's only ligand, or --binder, is probed with the default
pocket. Takes every option of boltz predict, plus --regions and --pocket_size.
"""

DEFAULT_REGIONS = 25
DEFAULT_POCKET_SIZE = 10
REGION_COLUMNS = (
    "chain",
    "residue",
    "residue_name",
    "region",
    "rank",
    "center_distance",
    "pocket",
)
# Occupancy of the region map marks the pocket residues of each region.
POCKET_OCCUPANCY = 1.0
OTHER_OCCUPANCY = 0.5


class Residue(NamedTuple):
    """A protein residue of the apo structure, placed by its CB atom (CA without)."""

    chain: str
    resid: int
    xyz: np.ndarray


class Region(NamedTuple):
    """One prediction: a region of the apo structure and the pocket made from it."""

    index: int  # 1-based, in sequence order of the regions' nearest residues
    center: np.ndarray
    members: list[tuple[str, int, float]]  # (chain, resid, distance), nearest first
    pocket: list[tuple[str, int]]  # the contacts: the nearest members


def parse_pocket_size(value: str) -> Optional[int]:
    """Read --pocket_size: a count of residues, or ``all`` (None) for every one."""
    text = str(value).strip().lower()
    if text == "all":
        return None
    if text.isdigit() and int(text) >= 1:
        return int(text)
    msg = f"{value}: expected a number of residues of at least 1, or all."
    raise click.BadParameter(msg, param_hint="--pocket_size")


def _mentions(value: object, chain: str) -> bool:
    """Whether a constraint or property refers to a chain, as binder or [chain, ...]."""
    if isinstance(value, dict):
        return any(
            (key == "binder" and str(item) == chain) or _mentions(item, chain)
            for key, item in value.items()
        )
    if isinstance(value, list):
        if value and isinstance(value[0], str) and value[0] == chain and 2 <= len(value) <= 3:  # noqa: PLR2004
            return True
        return any(_mentions(item, chain) for item in value)
    return False


def apo_schema(schema: dict, binder: str) -> dict:
    """Return the input without its binder: the structure the regions are read from.

    The binder's chain is removed from the sequences (its entry, or its id from
    an entry naming several), along with every constraint and property that
    refers to it. The other chains, such as cofactors, are kept.
    """
    apo = copy.deepcopy(schema)
    sequences = []
    for item in apo["sequences"]:
        (kind, entry), = item.items()
        ids = chain_ids(entry)
        if binder not in ids:
            sequences.append(item)
            continue
        if len(ids) > 1:
            entry["id"] = [i for i in ids if i != binder]
            sequences.append(item)
    apo["sequences"] = sequences
    for key in ("constraints", "properties"):
        kept = [c for c in apo.get(key) or [] if not _mentions(c, binder)]
        if kept:
            apo[key] = kept
        else:
            apo.pop(key, None)
    return apo


def read_residues(path: Path, chains: dict[str, dict], binder: str) -> list[Residue]:
    """Read each protein residue's CB atom from a structure, or its CA without one.

    Glycine has no CB, so it is placed by its CA. The binder, if a protein,
    is left out. Residues come in chain order of the input, then by number.
    """
    found: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for atom in read_atoms(path):
        if atom.chain in chains and atom.chain != binder and atom.name in ("CA", "CB"):
            found.setdefault((atom.chain, atom.resid), {})[atom.name] = atom.xyz
    order = {chain: index for index, chain in enumerate(chains)}
    return [
        Residue(chain, resid, atoms.get("CB", atoms.get("CA")))
        for (chain, resid), atoms in sorted(
            found.items(), key=lambda item: (order[item[0][0]], item[0][1])
        )
        if "CB" in atoms or "CA" in atoms
    ]


def cluster_regions(
    residues: list[Residue], k: int, pocket_size: Optional[int]
) -> list[Region]:
    """Split the residues into k regions by k-means on their CB atoms.

    Each region's pocket is its ``pocket_size`` residues nearest its center,
    or every residue of the region when ``pocket_size`` is None. The k-means
    uses a fixed random state, so the same structure always gives the same
    regions. Regions are numbered in sequence order of their nearest residue.
    """
    if len(residues) < k:
        msg = (
            f"The structure has {len(residues)} residues with a CB or CA atom, "
            f"fewer than the {k} regions asked for; lower --regions."
        )
        raise click.UsageError(msg)
    xyz = np.array([residue.xyz for residue in residues])
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(xyz)

    chain_order = {residue.chain: index for index, residue in enumerate(residues)}
    regions = []
    for label in range(k):
        indices = np.flatnonzero(kmeans.labels_ == label)
        center = kmeans.cluster_centers_[label]
        distances = np.linalg.norm(xyz[indices] - center, axis=1)
        nearest = indices[np.argsort(distances, kind="stable")]
        members = [
            (residues[i].chain, residues[i].resid, float(np.linalg.norm(xyz[i] - center)))
            for i in nearest
        ]
        pocket = [(chain, resid) for chain, resid, _ in members[:pocket_size]]
        regions.append((center, members, pocket))
    regions.sort(key=lambda r: (chain_order[r[1][0][0]], r[1][0][1]))
    return [
        Region(index, center, members, pocket)
        for index, (center, members, pocket) in enumerate(regions, start=1)
    ]


ResidueValue = Callable[[str, int], float]


def _annotate_mae(
    source: Path, target: Path, bfactor: ResidueValue, occupancy: ResidueValue
) -> None:
    """Copy a MAE file Boltz wrote, with other B-factors and an occupancy column.

    Boltz writes the B-factor (``r_m_pdb_tfactor``) as the last column of each
    atom row, so it is swapped there, and ``r_m_pdb_occupancy`` is added after
    it. Everything else in the file, the bonds and properties included, is kept.
    """
    out, state, columns = [], "before", []
    for line in source.read_text().splitlines():
        stripped = line.strip()
        if state == "before":
            out.append(line)
            if stripped.startswith("m_atom["):
                state = "columns"
        elif state == "columns":
            if stripped != ":::":
                columns.append(stripped)
                out.append(line)
                continue
            if not columns or columns[-1] != "r_m_pdb_tfactor":
                msg = f"{source.name}: expected r_m_pdb_tfactor as the last atom column."
                raise click.ClickException(msg)
            out.extend(["  r_m_pdb_occupancy", line])
            state = "rows"
        elif state == "rows":
            if stripped == ":::":
                out.append(line)
                state = "after"
                continue
            fields = shlex.split(line)
            chain = fields[columns.index("s_m_chain_name") + 1]
            resid = int(fields[columns.index("i_m_residue_number") + 1])
            head = line.rstrip().rsplit(None, 1)[0]
            out.append(f"{head} {bfactor(chain, resid):.2f} {occupancy(chain, resid):.2f}")
        else:
            out.append(line)
    target.write_text("\n".join(out) + "\n")


def _annotate_dms(
    source: Path, target: Path, bfactor: ResidueValue, occupancy: ResidueValue
) -> None:
    """Copy a DMS file, with other B-factors and an occupancy column."""
    shutil.copyfile(source, target)
    con = sqlite3.connect(target)
    try:
        with con:
            present = {row[1] for row in con.execute("pragma table_info(particle)")}
            for column in ("bfactor", "occupancy"):
                if column not in present:
                    con.execute(f"alter table particle add column {column} float")  # noqa: S608
            residues = con.execute("select distinct chain, resid from particle").fetchall()
            con.executemany(
                "update particle set bfactor = ?, occupancy = ? "
                "where chain = ? and resid = ?",
                [
                    (bfactor(str(c), int(r)), occupancy(str(c), int(r)), c, r)
                    for c, r in residues
                ],
            )
    finally:
        con.close()


def _annotate_gemmi(
    source: Path, target: Path, bfactor: ResidueValue, occupancy: ResidueValue
) -> None:
    """Copy a PDB or mmCIF file, with other B-factors and occupancies."""
    structure = gemmi.read_structure(str(source))
    for chain in structure[0]:
        for residue in chain:
            b = bfactor(chain.name, residue.seqid.num)
            q = occupancy(chain.name, residue.seqid.num)
            for atom in residue:
                atom.b_iso = b
                atom.occ = q
    if target.suffix == ".cif":
        structure.make_mmcif_document().write_file(str(target))
    else:
        structure.write_pdb(str(target))


def write_regions(
    results: Path,
    apo_structure: Path,
    regions: list[Region],
    chains: dict[str, dict],
    name: str,
) -> tuple[Path, Path, Optional[Path]]:
    """Write the regions: a CSV of residues, a map, and a PyMOL script.

    ``regions.csv`` lists every residue with its region and its rank by
    distance to the region's center. The map is the apo structure in its own
    format (``regions.mae`` for MAE output, and so on) with each residue's
    region in the B-factor (0 for other chains), and an occupancy of 1 for
    the pocket residues and 0.5 for the rest, so a viewer can color the
    regions and pick out the pockets. ``regions.pml`` does that in PyMOL, with
    a labelled sphere at each region's center; it is not written for DMS,
    which PyMOL does not read.
    """
    region_of, pocket = {}, set()
    rows = []
    for region in regions:
        pocket.update(region.pocket)
        for rank, (chain, resid, distance) in enumerate(region.members, start=1):
            region_of[chain, resid] = region.index
            rows.append(
                {
                    "chain": chain,
                    "residue": resid,
                    "residue_name": residue_name(chains[chain], resid),
                    "region": region.index,
                    "rank": rank,
                    "center_distance": round(distance, 3),
                    "pocket": (chain, resid) in pocket,
                }
            )
    rows.sort(key=lambda row: (row["region"], row["rank"]))
    # With --apo nothing has created the results folder yet.
    results.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(results / "regions.csv", rows, list(REGION_COLUMNS))

    def bfactor(chain: str, resid: int) -> float:
        return float(region_of.get((chain, resid), 0))

    def occupancy(chain: str, resid: int) -> float:
        return POCKET_OCCUPANCY if (chain, resid) in pocket else OTHER_OCCUPANCY

    map_path = results / f"regions{apo_structure.suffix}"
    annotate = {".mae": _annotate_mae, ".dms": _annotate_dms}.get(
        apo_structure.suffix, _annotate_gemmi
    )
    annotate(apo_structure, map_path, bfactor, occupancy)
    if apo_structure.suffix == ".dms":
        return csv_path, map_path, None

    k = len(regions)
    script = [
        f"# Regions of {name}: B-factor = region (1-{k}), occupancy 1 = pocket residue",
        f"load {map_path.name}, regions",
        "hide everything, regions",
        "show cartoon, regions",
        "color grey80, regions and b < 0.5",
        f"spectrum b, rainbow, regions and b > 0.5, minimum=1, maximum={k}",
        "show sticks, regions and q > 0.75 and not name N+C+O",
        "set cartoon_transparency, 0.3, regions",
    ]
    for region in regions:
        x, y, z = (round(float(c), 3) for c in region.center)
        script.append(
            f"pseudoatom centers, pos=[{x}, {y}, {z}], resi={region.index}, "
            f"b={region.index}, name=R{region.index}"
        )
    script += [
        "show spheres, centers",
        "set sphere_scale, 0.6, centers",
        f"spectrum b, rainbow, centers, minimum=1, maximum={k}",
        "label centers, resi",
        "set label_size, 20",
        "orient regions",
    ]
    pml_path = results / "regions.pml"
    pml_path.write_text("\n".join(script) + "\n")
    return csv_path, map_path, pml_path


def predict_apo(
    predict: click.Command,
    options: dict,
    results: Path,
    name: str,
    schema: dict,
    binder: str,
) -> Path:
    """Predict the input without its binder, and return its top-ranked structure.

    The run goes under the results folder, as ``boltz_results_apo``, with the
    same options as the probing run; a rerun finds it there and skips it.
    """
    record_id = f"{name}_apo"
    inputs_dir = results / "regionprobe_inputs" / "apo"
    write_inputs({record_id: apo_schema(schema, binder)}, inputs_dir)
    run_predict(predict, {**options, "out_dir": str(results)}, inputs_dir)
    folder = results / "boltz_results_apo" / "predictions" / record_id
    models = predicted_models(folder, record_id, str(options["output_format"]))
    structure = models[0][1] if models else None
    if structure is None:
        msg = f"No apo structure was written in {folder}; cannot find the regions."
        raise click.ClickException(msg)
    return structure


def center_distance(structure: Path, binder: str, center: np.ndarray) -> Optional[float]:
    """Return the distance from the binder's center of mass to a region's center."""
    ligand = [atom for atom in read_atoms(structure) if atom.chain == binder]
    if not ligand:
        return None
    xyz = np.array([atom.xyz for atom in ligand])
    masses = np.array([atom.mass for atom in ligand])
    com = np.average(xyz, axis=0, weights=masses if masses.sum() > 0 else None)
    return float(np.linalg.norm(com - center))


def _format_pocket(pocket: list[tuple[str, int]]) -> str:
    return ";".join(f"{chain}:{resid}" for chain, resid in pocket)


def write_summary(
    results: Path,
    records: dict[str, Region],
    template: dict,
    output_format: str,
    *,
    ligand_smiles: Optional[str] = None,
    affinity_binder: bool = False,
    bonds: Optional[list] = None,
) -> Path:
    """Tabulate every structure: its region, pocket, SMILES, distances and scores."""
    binder = str(template["binder"])
    max_distance = float(template.get("max_distance", DEFAULT_MAX_DISTANCE))

    rows = []
    for record_id, region in records.items():
        folder = results / "predictions" / record_id
        affinity = read_affinity(folder, record_id)
        for model, structure, confidence in predicted_models(
            folder, record_id, output_format
        ):
            row = {
                "region": region.index,
                "region_residues": len(region.members),
                "pocket_residues": _format_pocket(region.pocket),
                "model": model,
                "structure": structure.name if structure else "",
            }
            row.update(
                ligand_smiles_columns(structure, binder, ligand_smiles, affinity_binder)
            )
            row.update(
                distance_columns(structure, binder, region.pocket, max_distance)
            )
            distance = (
                center_distance(structure, binder, region.center)
                if structure is not None
                else None
            )
            row["region_center_distance"] = "" if distance is None else round(distance, 3)
            if bonds:
                row.update(bond_columns(structure, bonds))
            row.update(score_columns(confidence, affinity))
            rows.append(row)

    fields = [
        "region",
        "region_residues",
        "pocket_residues",
        "model",
        "structure",
        *SMILES_FIELDS,
        *DISTANCE_FIELDS,
        "region_center_distance",
        *(BOND_FIELDS if bonds else ()),
        *CONFIDENCE_SCORES,
    ]
    return write_csv(results / "regionprobe_summary.csv", rows, fields)


def make_regionprobe_command(
    predict: click.Command, compute_msa: Callable
) -> click.Command:
    """Build ``boltz regionprobe`` from ``boltz predict``, sharing its options."""
    regions_option = click.Option(
        ["--regions"],
        type=click.IntRange(min=1),
        default=DEFAULT_REGIONS,
        show_default=True,
        help=(
            "The number of regions to split the protein into, by k-means on its "
            "CB atoms (CA for glycine) across all its chains; one prediction each."
        ),
    )
    pocket_size_option = click.Option(
        ["--pocket_size"],
        type=str,
        default=str(DEFAULT_POCKET_SIZE),
        show_default=True,
        help=(
            "The residues in each pocket: the ones of the region nearest its "
            "center, up to this many, or all for every residue of the region."
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
    apo_option = click.Option(
        ["--apo"],
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help=(
            "A structure to read the regions from, instead of predicting the "
            "input without its binder: a Boltz output with the input's chain ids "
            "and residue numbers, such as an earlier run's "
            "boltz_results_apo/predictions/<input>_apo/<input>_apo_model_0.mae. "
            "Runs with other --pocket_size values then share the same regions."
        ),
    )

    def regionprobe(**options: object) -> None:  # noqa: PLR0915
        check_guidance_weights(options)
        k = int(options.pop("regions"))
        pocket_size = parse_pocket_size(str(options.pop("pocket_size")))
        binder_chain = options.pop("binder")
        apo = options.pop("apo")
        data = Path(str(options["data"])).expanduser()
        if data.suffix.lower() not in (".yml", ".yaml"):
            msg = "regionprobe takes a YAML file, which can hold pocket constraints."
            raise click.UsageError(msg)

        schema = yaml.safe_load(data.read_text())
        found = find_template(schema)
        if found is None:
            binder = choose_binder(schema, binder_chain)
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
        if not [chain for chain in chains if chain != binder]:
            msg = "There is no protein chain to find regions in."
            raise click.UsageError(msg)
        if options["model"] == "boltz1":
            check_boltz1(schema, template)

        # Another k or pocket size makes other pockets under the same names,
        # so each gets its own results.
        name = data.stem
        run = (
            f"{name}_regionprobe"
            + (f"_k{k}" if k != DEFAULT_REGIONS else "")
            + (
                ""
                if pocket_size == DEFAULT_POCKET_SIZE
                else "_mall"
                if pocket_size is None
                else f"_m{pocket_size}"
            )
        )
        results, inputs_dir = results_paths(options, run, "regionprobe_inputs")
        share_msa(schema, name, results / "msa", options, compute_msa)

        if apo is not None:
            apo_structure = Path(str(apo)).expanduser()
            click.echo(f"Regionprobe: reading the regions of {name} from {apo_structure}.")
        else:
            click.echo(f"Regionprobe: predicting {name} without {binder} to find its regions.")
            apo_structure = predict_apo(predict, options, results, name, schema, binder)
        residues = read_residues(apo_structure, chains, binder)
        found_chains = {residue.chain for residue in residues}
        absent = [c for c in chains if c != binder and c not in found_chains]
        if absent:
            msg = (
                f"{apo_structure.name} has no CA or CB atoms of chain(s) "
                f"{', '.join(absent)}; it must carry the input's chain ids."
            )
            raise click.UsageError(msg)
        regions = cluster_regions(residues, k, pocket_size)
        _, map_path, pml_path = write_regions(results, apo_structure, regions, chains, name)
        colored = f", colored by {pml_path.name}" if pml_path else ""
        click.echo(
            f"Found {k} regions among {len(residues)} residues of {apo_structure.name}; "
            f"map written to {map_path} (B-factor = region){colored}."
        )

        width = len(str(k))
        records, inputs = {}, {}
        for region in regions:
            record_id = f"{name}_region{region.index:0{width}d}"
            variant = copy.deepcopy(schema)
            variant["constraints"][template_index]["pocket"]["contacts"] = [
                [chain, resid] for chain, resid in region.pocket
            ]
            records[record_id] = region
            inputs[record_id] = variant
        write_inputs(inputs, inputs_dir)
        per_pocket = (
            "every residue of the region"
            if pocket_size is None
            else f"up to {pocket_size} residues"
        )
        click.echo(
            f"Regionprobe: probing {k} regions with {binder}, {per_pocket} per "
            f"pocket, {options['diffusion_samples']} structure(s) each."
        )

        run_predict(predict, options, inputs_dir)

        binder_entry = chains_of_kind(schema, "ligand").get(binder) or {}
        summary = write_summary(
            results,
            records,
            template,
            str(options["output_format"]),
            ligand_smiles=binder_entry.get("smiles"),
            affinity_binder=binder in affinity_binders(schema),
            bonds=bond_constraints(schema),
        )
        click.echo(f"Regionprobe summary written to {summary}.")

    return click.Command(
        name="regionprobe",
        callback=regionprobe,
        params=[
            *predict.params,
            regions_option,
            pocket_size_option,
            binder_option,
            apo_option,
        ],
        help=HELP,
        short_help="Probe every region of a protein as a pocket for a ligand.",
    )
