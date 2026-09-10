from typing import Optional

from rdkit import Chem
from rdkit.Chem import Mol
from torch import Tensor

from boltz.data.types import Structure
from boltz.data.write.topology import TopologyAtom, build_topology

_MAE_VERSION_BLOCK = "{\n s_m_m2io_version\n :::\n 2.0.0\n}\n\n"

_ATOM_COLUMNS = (
    "i_m_residue_number",
    "s_m_chain_name",
    "s_m_pdb_residue_name",
    "s_m_pdb_atom_name",
    "i_m_atomic_number",
    "i_m_formal_charge",
    "r_m_x_coord",
    "r_m_y_coord",
    "r_m_z_coord",
    "r_m_pdb_tfactor",
)

_BOND_COLUMNS = ("i_m_from", "i_m_to", "i_m_order")


def _quote(value: object) -> str:
    """Quote a MAE string value, escaping what the format reserves."""
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _pdb_atom_name(atom: TopologyAtom) -> str:
    """Pad an atom name into its four-character PDB column.

    ``s_m_pdb_atom_name`` is the PDB field, so a one-letter element is indented
    by one space exactly as a PDB writes it; an unpadded name puts the element
    in the wrong column for anything that reads the padding back.
    """
    name = atom.name[:4]
    symbol = (
        Chem.GetPeriodicTable().GetElementSymbol(atom.atomic_number)
        if atom.atomic_number
        else ""
    )
    if len(name) < 4 and name[:1].isalpha() and len(symbol) < 2:  # noqa: PLR2004
        name = " " + name
    return f"{name:<4}"


def to_mae(
    structure: Structure,
    molecules: dict[str, Mol],
    plddts: Optional[Tensor] = None,
    boltz2: bool = False,
    title: str = "boltz",
) -> str:
    """Write a structure into a Maestro file.

    Unlike PDB and mmCIF output, every bond is written with its order, and
    every atom with its formal charge. The pLDDT goes in the B-factor.

    Parameters
    ----------
    structure : Structure
        The input structure
    molecules : dict[str, Mol]
        Reference molecules by residue name.
    title : str
        The title of the structure.

    Returns
    -------
    str
        the output MAE file

    """
    topology = build_topology(structure, molecules, plddts=plddts, boltz2=boltz2)

    lines = [_MAE_VERSION_BLOCK + "f_m_ct {", " s_m_title", " :::", f" {_quote(title)}"]

    lines.append(f" m_atom[{len(topology.atoms)}] {{")
    lines.extend(f"  {column}" for column in _ATOM_COLUMNS)
    lines.append("  :::")
    for index, atom in enumerate(topology.atoms, start=1):
        x, y, z = atom.coords
        lines.append(
            f"  {index} {atom.res_id} {_quote(atom.chain)} "
            f"{_quote(f'{atom.res_name[:4]:<4}')} {_quote(_pdb_atom_name(atom))} "
            f"{atom.atomic_number} {atom.formal_charge} "
            f"{x:.6f} {y:.6f} {z:.6f} {atom.bfactor:.2f}"
        )
    lines.extend(["  :::", " }"])

    if topology.bonds:
        lines.append(f" m_bond[{len(topology.bonds)}] {{")
        lines.extend(f"  {column}" for column in _BOND_COLUMNS)
        lines.append("  :::")
        for index, (atom_1, atom_2, order) in enumerate(topology.bonds, start=1):
            lines.append(f"  {index} {atom_1 + 1} {atom_2 + 1} {order}")
        lines.extend(["  :::", " }"])

    lines.extend(["}", ""])
    return "\n".join(lines)
