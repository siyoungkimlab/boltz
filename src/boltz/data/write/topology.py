"""Atom and bond tables for the writers that keep bond orders and charges.

PDB and mmCIF output drop the chemistry Boltz carries internally: bond orders,
formal charges, and most bonds. The MAE and DMS writers store all three, and
build their tables here from the predicted structure and the reference
molecules the featurizer read those atoms from.
"""

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from torch import Tensor

from boltz.data import const
from boltz.data.types import Structure

# Bond orders by RDKit bond type once aromatic rings are kekulized. Anything
# else (dative, or an aromatic bond that would not kekulize) is single.
_RDKIT_ORDERS = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}

# The same orders by Boltz bond type id; aromatic, covalent and other are single.
_BOLTZ_ORDERS = {
    const.bond_type_ids[name]: order for name, order in _RDKIT_ORDERS.items()
}

# The atoms that join a polymer residue to the next one in its chain. Boltz
# stores no bonds for standard residues, so these are added here.
_POLYMER_LINKS = {
    const.chain_type_ids["PROTEIN"]: ("C", "N"),
    const.chain_type_ids["DNA"]: ("O3'", "P"),
    const.chain_type_ids["RNA"]: ("O3'", "P"),
}


@dataclass
class TopologyAtom:
    """One written atom."""

    name: str
    atomic_number: int
    formal_charge: int
    coords: tuple[float, float, float]
    res_name: str
    res_id: int
    chain: str
    bfactor: float


@dataclass
class Topology:
    """Atoms in file order, and bonds as zero-based (atom, atom, order)."""

    atoms: list[TopologyAtom]
    bonds: list[tuple[int, int, int]]


@dataclass
class _Template:
    """A residue's reference chemistry, keyed by atom name."""

    atoms: dict[str, tuple[int, int]]  # name -> (atomic number, formal charge)
    bonds: dict[tuple[str, str], int]  # (name, name) -> bond order


def _template(res_name: str, mol: Mol) -> _Template:
    """Read atomic numbers, charges and Kekule bond orders off a reference mol."""
    mol = Chem.Mol(mol)
    # A SMILES ligand is stored unsanitized; without valences, kekulization
    # would saturate its aromatic rings instead of alternating them.
    mol.UpdatePropertyCache(strict=False)
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Chem.KekulizeException:
        print(f"Could not kekulize {res_name}, writing its aromatic bonds as single.")  # noqa: T201

    atoms = {}
    names = {}
    for atom in mol.GetAtoms():
        if not atom.HasProp("name"):
            continue
        name = atom.GetProp("name")
        atoms[name] = (atom.GetAtomicNum(), atom.GetFormalCharge())
        names[atom.GetIdx()] = name

    bonds = {}
    for bond in mol.GetBonds():
        name_1 = names.get(bond.GetBeginAtomIdx())
        name_2 = names.get(bond.GetEndAtomIdx())
        if name_1 is not None and name_2 is not None:
            bonds[(name_1, name_2)] = _RDKIT_ORDERS.get(bond.GetBondType().name, 1)
    return _Template(atoms, bonds)


# Protein residues whose reference molecule is not in its state at pH 7: the
# CCD holds aspartate and glutamate as neutral acids and histidine protonated.
# Each gets its pH 7 formal charges here, with the bond orders that go with
# them, histidine as the neutral NE2 tautomer (HIE). Lysine and arginine
# already carry their +1.
_PROTEIN_STATES = {
    "ASP": ({"OD1": 0, "OD2": -1}, {("CG", "OD1"): 2, ("CG", "OD2"): 1}),
    "GLU": ({"OE1": 0, "OE2": -1}, {("CD", "OE1"): 2, ("CD", "OE2"): 1}),
    "HIS": (
        {"ND1": 0, "NE2": 0},
        {
            ("CG", "CD2"): 2,
            ("CG", "ND1"): 1,
            ("ND1", "CE1"): 2,
            ("CE1", "NE2"): 1,
            ("NE2", "CD2"): 1,
        },
    ),
}


def _protein_state(res_name: str, template: _Template) -> _Template:
    """Put a protein residue's template in its pH 7 state."""
    if res_name not in _PROTEIN_STATES:
        return template
    charges, orders = _PROTEIN_STATES[res_name]
    atoms = dict(template.atoms)
    for name, charge in charges.items():
        if name in atoms:
            atoms[name] = (atoms[name][0], charge)
    bonds = dict(template.bonds)
    for (name_1, name_2), order in orders.items():
        key = (name_1, name_2) if (name_1, name_2) in bonds else (name_2, name_1)
        if key in bonds:
            bonds[key] = order
    return _Template(atoms, bonds)


def _atom_name(atom: np.void, boltz2: bool) -> str:
    if boltz2:
        return str(atom["name"])
    return "".join(chr(c + 32) for c in atom["name"] if c != 0)


def _element_from_name(atom_name: str, res_name: str) -> int:
    """Guess the atomic number from the atom name, as the PDB writer does."""
    atom_key = re.sub(r"\d", "", atom_name)
    if atom_key in const.ambiguous_atoms:
        ambiguous = const.ambiguous_atoms[atom_key]
        if isinstance(ambiguous, str):
            element = ambiguous
        elif res_name in ambiguous:
            element = ambiguous[res_name]
        else:
            element = ambiguous["*"]
    else:
        element = atom_key[0]
    return Chem.GetPeriodicTable().GetAtomicNumber(element.capitalize())


def build_topology(  # noqa: C901, PLR0912, PLR0915
    structure: Structure,
    molecules: dict[str, Mol],
    plddts: Optional[Tensor] = None,
    boltz2: bool = False,
) -> Topology:
    """Collect the atoms and bonds of a structure with their chemistry.

    Each residue takes its atomic numbers, formal charges and bond orders from
    its reference molecule in ``molecules``, matched by atom name. A residue
    without one falls back to what the structure stores: its bonds (aromatic as
    single) and, for Boltz-1, its elements and charges.

    Parameters
    ----------
    structure : Structure
        The predicted structure.
    molecules : dict[str, Mol]
        Reference molecules by residue name.
    plddts : Tensor, optional
        Per-token pLDDT, written as the B-factor.
    boltz2 : bool
        Whether the structure is a Boltz-2 ``StructureV2``.

    Returns
    -------
    Topology
        The atoms and bonds to write.

    """
    # Templates by residue name and whether the residue is in a protein chain.
    templates: dict[tuple[str, bool], Optional[_Template]] = {}
    atoms: list[TopologyAtom] = []
    bonds: dict[tuple[int, int], int] = {}
    index = {}  # structure atom index -> written atom index
    residue_atoms = {}  # structure residue index -> {atom name: written index}

    def add_bond(atom_1: int, atom_2: int, order: int) -> None:
        # The first order recorded wins, so a template's is never overwritten.
        bonds.setdefault((min(atom_1, atom_2), max(atom_1, atom_2)), order)

    # Index into the plddt tensor, tracked exactly as the PDB writer does.
    res_num = 0
    prev_polymer_resnum = -1
    ligand_index_offset = 0

    for chain in structure.chains:
        het = chain["mol_type"] == const.chain_type_ids["NONPOLYMER"]
        protein = chain["mol_type"] == const.chain_type_ids["PROTEIN"]
        res_start = chain["res_idx"]
        res_end = chain["res_idx"] + chain["res_num"]
        for res_idx in range(res_start, res_end):
            residue = structure.residues[res_idx]
            res_name = str(residue["name"])
            key = (res_name, protein)
            if key not in templates:
                mol = molecules.get(res_name)
                if mol is None:
                    print(  # noqa: T201
                        f"No reference molecule for {res_name}, writing it from "
                        "the structure: aromatic bonds as single, and for "
                        "Boltz-2, charges as zero."
                    )
                template = None if mol is None else _template(res_name, mol)
                if template is not None and protein:
                    template = _protein_state(res_name, template)
                templates[key] = template
            template = templates[key]

            names = {}
            atom_start = residue["atom_idx"]
            atom_end = residue["atom_idx"] + residue["atom_num"]
            for atom_idx in range(atom_start, atom_end):
                atom = structure.atoms[atom_idx]
                # This should not happen on predictions, but just in case.
                if not atom["is_present"]:
                    continue

                if not het:
                    bfactor = (
                        100.00
                        if plddts is None
                        else round(
                            plddts[res_num + ligand_index_offset].item() * 100, 2
                        )
                    )
                    prev_polymer_resnum = res_num
                else:
                    ligand_index_offset += 1
                    bfactor = (
                        100.00
                        if plddts is None
                        else round(
                            plddts[prev_polymer_resnum + ligand_index_offset].item()
                            * 100,
                            2,
                        )
                    )

                name = _atom_name(atom, boltz2)
                if template is not None and name in template.atoms:
                    atomic_number, formal_charge = template.atoms[name]
                elif boltz2:
                    atomic_number, formal_charge = _element_from_name(name, res_name), 0
                else:
                    atomic_number, formal_charge = (
                        int(atom["element"]),
                        int(atom["charge"]),
                    )

                index[atom_idx] = len(atoms)
                names[name] = len(atoms)
                atoms.append(
                    TopologyAtom(
                        name=name,
                        atomic_number=atomic_number,
                        formal_charge=formal_charge,
                        coords=tuple(float(c) for c in atom["coords"]),
                        res_name=res_name,
                        res_id=int(residue["res_idx"]) + 1,
                        chain=str(chain["name"]),
                        bfactor=bfactor,
                    )
                )

            residue_atoms[res_idx] = names
            if template is not None:
                for (name_1, name_2), order in template.bonds.items():
                    if name_1 in names and name_2 in names:
                        add_bond(names[name_1], names[name_2], order)

            if not het:
                res_num += 1

    # Bonds the structure stores: ligand and modified residue bonds, and the
    # covalent connections from the input constraints.
    stored = [
        (bond["atom_1"], bond["atom_2"], _BOLTZ_ORDERS.get(int(bond["type"]), 1))
        for bond in structure.bonds
    ]
    if not boltz2:
        stored += [(c["atom_1"], c["atom_2"], 1) for c in structure.connections]
    for atom_1, atom_2, order in stored:
        if atom_1 in index and atom_2 in index:
            add_bond(index[atom_1], index[atom_2], order)

    # Backbone links between consecutive residues, closing cyclic chains.
    for chain in structure.chains:
        link = _POLYMER_LINKS.get(int(chain["mol_type"]))
        if link is None:
            continue
        res_start = int(chain["res_idx"])
        res_end = res_start + int(chain["res_num"])
        pairs = [(i, i + 1) for i in range(res_start, res_end - 1)]
        if chain["cyclic_period"] > 0 and res_end - res_start > 1:
            pairs.append((res_end - 1, res_start))
        for res_1, res_2 in pairs:
            atom_1 = residue_atoms[res_1].get(link[0])
            atom_2 = residue_atoms[res_2].get(link[1])
            if atom_1 is not None and atom_2 is not None:
                add_bond(atom_1, atom_2, 1)

    return Topology(
        atoms=atoms,
        bonds=[
            (atom_1, atom_2, order) for (atom_1, atom_2), order in sorted(bonds.items())
        ],
    )
