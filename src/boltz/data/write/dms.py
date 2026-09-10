import sqlite3
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Mol
from torch import Tensor

from boltz.data.types import Structure
from boltz.data.write.topology import build_topology


def to_dms(
    path: Path,
    structure: Structure,
    molecules: dict[str, Mol],
    plddts: Optional[Tensor] = None,
    boltz2: bool = False,
) -> None:
    """Write a structure into a DMS file.

    DMS is a SQLite database rather than text, so this writes ``path`` itself.
    Every bond is written with its order, and every atom with its formal
    charge. The pLDDT goes in the ``bfactor`` column.

    Parameters
    ----------
    path : Path
        The file to write, replaced if it exists.
    structure : Structure
        The input structure
    molecules : dict[str, Mol]
        Reference molecules by residue name.

    """
    topology = build_topology(structure, molecules, plddts=plddts, boltz2=boltz2)
    periodic_table = Chem.GetPeriodicTable()

    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    try:
        with con:
            con.execute("create table dms_version (major integer, minor integer)")
            con.execute("insert into dms_version values (1, 7)")
            con.execute(
                "create table particle (id integer primary key, anum integer, "
                "name text, x float, y float, z float, vx float, vy float, "
                "vz float, resname text, resid integer, insertion text, "
                "chain text, segid text, mass float, charge float, "
                "formal_charge integer, bfactor float)"
            )
            con.executemany(
                "insert into particle values "
                "(?, ?, ?, ?, ?, ?, 0, 0, 0, ?, ?, '', ?, '', ?, 0, ?, ?)",
                [
                    (
                        index,
                        atom.atomic_number,
                        atom.name,
                        *atom.coords,
                        atom.res_name,
                        atom.res_id,
                        atom.chain,
                        periodic_table.GetAtomicWeight(atom.atomic_number)
                        if atom.atomic_number
                        else 0.0,
                        atom.formal_charge,
                        atom.bfactor,
                    )
                    for index, atom in enumerate(topology.atoms)
                ],
            )
            con.execute('create table bond (p0 integer, p1 integer, "order" integer)')
            con.executemany("insert into bond values (?, ?, ?)", topology.bonds)
            # No periodic cell: a prediction is not a simulation box.
            con.execute(
                "create table global_cell (id integer primary key, "
                "x float, y float, z float)"
            )
            con.executemany(
                "insert into global_cell values (?, 0, 0, 0)", [(0,), (1,), (2,)]
            )
    finally:
        con.close()
