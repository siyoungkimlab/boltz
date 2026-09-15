"""Shared pieces of commands that run boltz predict on many generated inputs.

``boltz pointprobe`` and ``boltz screen`` both turn one input YAML into many,
one per residue or per ligand, and predict them all in a single ``boltz
predict`` run: the MSA is computed once and the model is loaded once. This
module holds what they share: finding chains, sharing the MSA, writing the
inputs, reading the predictions back, and writing their summary.
"""

import csv
import hashlib
import json
import shlex
import sqlite3
from pathlib import Path
from typing import Callable, NamedTuple, Optional

import click
import gemmi
import numpy as np
import yaml
from rdkit import Chem
from rdkit.Geometry import Point3D

from boltz.atomname import boltz_atom_names, heavy_atom_names, named_smiles
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
DISTANCE_FIELDS = (*DISTANCES, "within_max_distance")
SMILES_FIELDS = ("input_smiles", "predicted_smiles", "matches_input")
STRUCTURE_SUFFIXES = {"mae": ".mae", "dms": ".dms", "pdb": ".pdb", "mmcif": ".cif"}
DEFAULT_MAX_DISTANCE = 6.0


def chain_ids(entry: dict) -> list[str]:
    """Return the chain ids of a sequence entry, which may name several."""
    ids = entry["id"]
    return [str(i) for i in ids] if isinstance(ids, list) else [str(ids)]


def chains_of_kind(schema: dict, kind: str) -> dict[str, dict]:
    """Map each chain id of one kind, such as protein, to its YAML entry."""
    chains = {}
    for item in schema["sequences"]:
        for item_kind, entry in item.items():
            if item_kind == kind:
                for chain in chain_ids(entry):
                    chains[chain] = entry
    return chains


def all_chain_ids(schema: dict) -> set[str]:
    """Return every chain id of the input."""
    return {
        chain
        for item in schema["sequences"]
        for entry in item.values()
        for chain in chain_ids(entry)
    }


def affinity_binders(schema: dict) -> set[str]:
    """Return the chain ids of the input's affinity binders."""
    return {
        str(prop["affinity"]["binder"])
        for prop in schema.get("properties") or []
        if "affinity" in prop
    }


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


def results_paths(options: dict, run: str, inputs_folder: str) -> tuple[Path, Path]:
    """Return the results folder of a run, and the folder for its inputs.

    ``boltz predict`` names its results folder after its folder of inputs, so
    the inputs go in ``<results>/<inputs_folder>/<run>`` for the results to be
    ``boltz_results_<run>``, holding the inputs too.
    """
    out_dir = Path(str(options["out_dir"])).expanduser()
    results = out_dir / f"boltz_results_{run}"
    return results, results / inputs_folder / run


def write_inputs(inputs: dict[str, dict], inputs_dir: Path) -> None:
    """Write each input YAML under its record id, replacing any from before."""
    inputs_dir.mkdir(parents=True, exist_ok=True)
    # A rerun may predict other inputs, so start from an empty folder.
    for old in inputs_dir.glob("*.yaml"):
        old.unlink()
    for record_id, schema in inputs.items():
        with (inputs_dir / f"{record_id}.yaml").open("w") as f:
            yaml.safe_dump(schema, f, sort_keys=False)


def run_predict(predict: click.Command, options: dict, inputs_dir: Path) -> None:
    """Predict every input in the folder, in one ``boltz predict`` run."""
    options["data"] = str(inputs_dir)
    click.get_current_context().invoke(predict, **options)


class StructureAtom(NamedTuple):
    """An atom of a predicted structure, as far as the distances need it."""

    chain: str
    resid: int
    name: str
    mass: float
    xyz: np.ndarray


def _mae_rows(text: str, table: str) -> list[dict[str, str]]:
    """Read the rows of one table of a Boltz MAE file, such as ``m_atom``."""
    if f" {table}[" not in text:
        return []
    block = text.split(f" {table}[", 1)[1].split(":::\n")
    columns = [c.strip() for c in block[0].splitlines()[1:] if c.strip()]
    return [
        dict(zip(["index", *columns], shlex.split(line)))
        for line in block[1].splitlines()
        if line.strip()
    ]


def read_atoms(path: Path) -> list[StructureAtom]:
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
            StructureAtom(str(c), int(r), str(n).strip(), float(m), np.array(xyz))
            for c, r, n, m, *xyz in rows
        ]

    if path.suffix == ".mae":
        periodic_table = Chem.GetPeriodicTable()
        atoms = []
        for row in _mae_rows(path.read_text(), "m_atom"):
            atomic_number = int(row["i_m_atomic_number"])
            atoms.append(
                StructureAtom(
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
        StructureAtom(
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


def _center_of_mass(atoms: list[StructureAtom]) -> np.ndarray:
    xyz = np.array([atom.xyz for atom in atoms])
    masses = np.array([atom.mass for atom in atoms])
    return np.average(xyz, axis=0, weights=masses if masses.sum() > 0 else None)


def contact_distances(
    path: Path, binder: str, contacts: list[tuple[str, int]]
) -> Optional[dict[str, Optional[float]]]:
    """Measure how close the binder came to its contact residues.

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
        for chain, resid in contacts
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


def distance_columns(
    structure: Optional[Path],
    binder: str,
    contacts: list[tuple[str, int]],
    max_distance: float,
) -> dict:
    """Return the distance columns of a summary row, blank where unmeasured."""
    distances = (
        contact_distances(structure, binder, contacts)
        if structure is not None and contacts
        else None
    ) or {}
    row = {}
    for key in DISTANCES:
        value = distances.get(key)
        row[key] = "" if value is None else round(value, 3)
    row["within_max_distance"] = (
        ""
        if row["max_contact_distance"] == ""
        else row["max_contact_distance"] <= max_distance
    )
    return row


_BOND_TYPES = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}


def read_ligand(path: Path, chain: str) -> Optional[Chem.Mol]:
    """Read one chain of a MAE or DMS structure as a molecule, in 3D.

    These formats keep every atom's formal charge and every bond's order, so the
    molecule is the chemistry the file holds; the hydrogens, which the output
    leaves out, are implicit. Its stereochemistry is read from the coordinates.
    Returns None for other formats, or if RDKit cannot sanitize the molecule.
    """
    if path.suffix == ".dms":
        con = sqlite3.connect(path)
        try:
            atoms = con.execute(
                "select id, name, anum, formal_charge, x, y, z from particle "
                "where chain = ? order by id",
                (chain,),
            ).fetchall()
            bonds = con.execute('select p0, p1, "order" from bond').fetchall()
        finally:
            con.close()
    elif path.suffix == ".mae":
        text = path.read_text()
        atoms = [
            (
                int(row["index"]),
                row["s_m_pdb_atom_name"],
                int(row["i_m_atomic_number"]),
                int(row["i_m_formal_charge"]),
                *(float(row[f"r_m_{axis}_coord"]) for axis in "xyz"),
            )
            for row in _mae_rows(text, "m_atom")
            if row["s_m_chain_name"] == chain
        ]
        bonds = [
            (int(row["i_m_from"]), int(row["i_m_to"]), int(row["i_m_order"]))
            for row in _mae_rows(text, "m_bond")
        ]
    else:
        return None
    if not atoms:
        return None

    mol = Chem.RWMol()
    conformer = Chem.Conformer(len(atoms))
    index = {}  # file atom id -> molecule atom index
    for file_id, name, atomic_number, charge, *xyz in atoms:
        atom = Chem.Atom(int(atomic_number))
        atom.SetFormalCharge(int(charge))
        atom.SetProp("name", str(name).strip())
        index[file_id] = mol.AddAtom(atom)
        conformer.SetAtomPosition(index[file_id], Point3D(*(float(c) for c in xyz)))
    for atom_1, atom_2, order in bonds:
        if atom_1 in index and atom_2 in index:
            mol.AddBond(
                index[atom_1],
                index[atom_2],
                _BOND_TYPES.get(int(order), Chem.BondType.SINGLE),
            )
    mol = mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except (ValueError, RuntimeError):
        return None
    mol.AddConformer(conformer, assignId=True)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def _placed_on_input(reference: Chem.Mol, path: Path, chain: str) -> Optional[Chem.Mol]:
    """Place the input's molecule on the predicted coordinates of its atoms.

    For PDB and mmCIF, which do not keep bond orders or charges: the chemistry
    is the input's, and only the stereochemistry comes from the prediction.
    """
    positions = {
        atom.name: atom.xyz for atom in read_atoms(path) if atom.chain == chain
    }
    mol = Chem.Mol(reference)
    mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for atom in mol.GetAtoms():
        xyz = positions.get(atom.GetProp("name"))
        if xyz is None:
            return None
        conformer.SetAtomPosition(atom.GetIdx(), Point3D(*(float(c) for c in xyz)))
    mol.AddConformer(conformer, assignId=True)
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def _matches_input(reference: Chem.Mol, predicted: Chem.Mol) -> bool:
    """Whether the prediction is the input molecule, as far as the input says.

    The same atoms, bonds and charges, and every stereocenter and double bond
    geometry the input specifies; those it leaves unspecified are ignored.
    Atoms are matched by the names Boltz gives them.
    """
    names = {atom.GetProp("name"): atom.GetIdx() for atom in predicted.GetAtoms()}
    order = [names.get(atom.GetProp("name")) for atom in reference.GetAtoms()]
    if None in order or len(order) != predicted.GetNumAtoms():
        return False
    check = Chem.RenumberAtoms(predicted, order)
    for ref_atom in reference.GetAtoms():
        if ref_atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
            check.GetAtomWithIdx(ref_atom.GetIdx()).SetChiralTag(
                Chem.ChiralType.CHI_UNSPECIFIED
            )
    for ref_bond in reference.GetBonds():
        bond = check.GetBondBetweenAtoms(
            ref_bond.GetBeginAtomIdx(), ref_bond.GetEndAtomIdx()
        )
        if bond is not None and ref_bond.GetStereo() == Chem.BondStereo.STEREONONE:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
    return Chem.MolToSmiles(check) == Chem.MolToSmiles(reference)


def ligand_smiles_columns(
    structure: Optional[Path],
    chain: str,
    smiles: Optional[str],
    affinity: bool = False,
) -> dict:
    """Return a ligand's input and predicted SMILES, and whether they agree.

    From MAE and DMS, the predicted SMILES is the ligand as the file holds it:
    its atoms, charges and bond orders, with stereochemistry from its predicted
    coordinates. PDB and mmCIF keep no bond orders or charges, so there the
    input's chemistry is placed on the predicted coordinates, matched by the
    names Boltz gives the atoms. The affinity binder is compared as Boltz
    standardizes it.
    """
    row = {"input_smiles": smiles or "", "predicted_smiles": "", "matches_input": ""}
    if structure is None or not smiles:
        return row
    reference = Chem.RemoveHs(boltz_atom_names(named_smiles(smiles, affinity)))
    if structure.suffix in (".mae", ".dms"):
        predicted = read_ligand(structure, chain)
    else:
        predicted = _placed_on_input(reference, structure, chain)
    if predicted is None:
        return row
    row["predicted_smiles"] = Chem.MolToSmiles(predicted)
    row["matches_input"] = _matches_input(reference, predicted)
    return row


def predicted_models(
    folder: Path, record_id: str, output_format: str
) -> list[tuple[int, Optional[Path], dict]]:
    """List a record's models: rank, structure file if written, confidence."""
    suffix = STRUCTURE_SUFFIXES.get(output_format)
    confidences = sorted(
        folder.glob(f"confidence_{record_id}_model_*.json"),
        key=lambda p: int(p.stem.rsplit("_", 1)[1]),
    )
    models = []
    for path in confidences:
        model = int(path.stem.rsplit("_", 1)[1])
        structure = folder / f"{record_id}_model_{model}{suffix}" if suffix else None
        if structure is not None and not structure.exists():
            structure = None
        models.append((model, structure, json.loads(path.read_text())))
    return models


def score_columns(confidence: dict, affinity: dict) -> dict:
    """Return the confidence and affinity columns of a summary row."""
    row = {key: confidence.get(key, "") for key in CONFIDENCE_SCORES}
    row.update({key: affinity.get(key, "") for key in AFFINITY_SCORES})
    return row


def read_affinity(folder: Path, record_id: str) -> dict:
    """Read a record's affinity prediction, if it has one."""
    path = folder / f"affinity_{record_id}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> Path:
    """Write a summary, with the affinity columns when any row has them."""
    if any(row.get(key, "") != "" for row in rows for key in AFFINITY_SCORES):
        fields = [*fields, *AFFINITY_SCORES]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


_LETTER_TO_RESIDUE = {
    "protein": const.prot_letter_to_token,
    "dna": const.dna_letter_to_token,
    "rna": const.rna_letter_to_token,
}


def _bond_atom_problem(  # noqa: PLR0911
    spec: object, entries: dict[str, tuple[str, dict]], affinity_binders: set[str]
) -> Optional[str]:
    """Say what is wrong with one atom of a bond constraint, if anything."""
    if not isinstance(spec, list) or len(spec) != 3:  # noqa: PLR2004
        return "expected [CHAIN, RESIDUE, ATOM]"
    chain, resid, name = str(spec[0]), spec[1], str(spec[2])
    if chain not in entries:
        return f"there is no chain {chain}"
    kind, entry = entries[chain]

    if kind == "ligand":
        if resid != 1:
            return f"a ligand is residue 1, not {resid}"
        if "smiles" not in entry:
            return None  # a CCD ligand's atoms are left to Boltz
        names = heavy_atom_names(str(entry["smiles"]), chain in affinity_binders)
        if name not in names:
            return (
                f"ligand {chain} has no atom {name}; its atoms are "
                f"{' '.join(names)} (see boltz atomname)"
            )
        return None

    sequence = str(entry.get("sequence", ""))
    if not isinstance(resid, int) or not 1 <= resid <= len(sequence):
        return f"chain {chain} has residues 1-{len(sequence)}"
    modified = {int(m["position"]) for m in entry.get("modifications") or []}
    if resid in modified:
        return None  # a modified residue's atoms are left to Boltz
    residue = _LETTER_TO_RESIDUE.get(kind, {}).get(sequence[resid - 1])
    atoms = const.ref_atoms.get(residue) if residue else None
    if atoms is not None and name not in atoms:
        return (
            f"residue {resid} of chain {chain} is {residue}, which has no atom "
            f"{name}; it has {' '.join(atoms)}"
        )
    return None


def check_bonds(schema: dict) -> None:
    """Check that every bond constraint's atoms exist, before running.

    A misnamed atom would otherwise fail every input in preprocessing. A SMILES
    ligand's atoms are checked against the names Boltz gives it, and a
    residue's against its standard atoms; the atoms of CCD ligands and of
    modified residues are left to Boltz.
    """
    entries = {
        chain: (kind, entry)
        for item in schema["sequences"]
        for kind, entry in item.items()
        for chain in chain_ids(entry)
    }
    binders = affinity_binders(schema)
    problems = []
    for constraint in schema.get("constraints") or []:
        if "bond" not in constraint:
            continue
        for key in ("atom1", "atom2"):
            spec = constraint["bond"].get(key)
            problem = _bond_atom_problem(spec, entries, binders)
            if problem:
                problems.append(f"bond {key} {spec}: {problem}")
    if problems:
        raise click.UsageError("\n".join(problems))
