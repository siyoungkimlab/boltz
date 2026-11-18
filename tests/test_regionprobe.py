"""Tests of boltz regionprobe's pieces that need no model: the apo input, the regions."""

from pathlib import Path

import click
import numpy as np
import pytest

from boltz.batch import _mae_rows
from boltz.regionprobe import (
    Residue,
    apo_schema,
    cluster_regions,
    parse_pocket_size,
    read_residues,
    write_regions,
)

SCHEMA = {
    "sequences": [
        {"protein": {"id": ["A", "B"], "sequence": "GAGA"}},
        {"ligand": {"id": ["L", "M"], "smiles": "CCO"}},
        {"ligand": {"id": "N", "smiles": "CO"}},
    ],
    "constraints": [
        {"pocket": {"binder": "L"}},
        {"bond": {"atom1": ["A", 2, "CB"], "atom2": ["L", 1, "C1"]}},
        {"contact": {"token1": ["A", 1], "token2": ["N", 1]}},
    ],
    "properties": [{"affinity": {"binder": "L"}}],
}


def test_apo_schema_drops_the_binder_and_what_refers_to_it() -> None:
    apo = apo_schema(SCHEMA, "L")
    assert apo["sequences"][1]["ligand"]["id"] == ["M"]
    assert [list(c)[0] for c in apo["constraints"]] == ["contact"]
    assert "properties" not in apo
    # The binder alone in its entry removes the entry.
    assert [list(s)[0] for s in apo_schema(SCHEMA, "N")["sequences"]] == [
        "protein",
        "ligand",
    ]


def test_parse_pocket_size() -> None:
    assert parse_pocket_size("10") == 10
    assert parse_pocket_size("All") is None
    with pytest.raises(click.BadParameter):
        parse_pocket_size("0")


def _blobs() -> list[Residue]:
    # Two blobs 50 A apart, chain A then B, the nearest residue of each first.
    residues = []
    for chain, origin in (("A", (0.0, 0.0, 0.0)), ("B", (50.0, 0.0, 0.0))):
        for resid, offset in enumerate(((0, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 3)), 1):
            residues.append(Residue(chain, resid, np.array(origin) + np.array(offset)))
    return residues


def test_cluster_regions_takes_the_nearest_residues_as_the_pocket() -> None:
    regions = cluster_regions(_blobs(), 2, pocket_size=2)
    assert [r.index for r in regions] == [1, 2]
    assert {c for c, _ in regions[0].pocket} == {"A"}
    assert {c for c, _ in regions[1].pocket} == {"B"}
    assert len(regions[0].pocket) == 2
    assert [m[1] for m in regions[0].members][:2] == [1, 2]
    # all: every residue of the region
    assert len(cluster_regions(_blobs(), 2, pocket_size=None)[0].pocket) == 4
    with pytest.raises(click.UsageError):
        cluster_regions(_blobs(), 9, pocket_size=2)


# A MAE file as boltz writes it: the pLDDT in r_m_pdb_tfactor, last.
_ATOMS = (
    (1, "A", "GLY", " N  ", 7, 0.0, 0.0, 0.0),
    (1, "A", "GLY", " CA ", 6, 1.0, 0.0, 0.0),
    (2, "A", "ALA", " N  ", 7, 5.0, 0.0, 0.0),
    (2, "A", "ALA", " CA ", 6, 6.0, 0.0, 0.0),
    (2, "A", "ALA", " CB ", 6, 7.0, 0.0, 0.0),
    (1, "B", "ALA", " CA ", 6, 50.0, 0.0, 0.0),
    (1, "B", "ALA", " CB ", 6, 51.0, 0.0, 0.0),
    (1, "L", "LIG ", " C1 ", 6, 20.0, 0.0, 0.0),
)
MAE = (
    "{\n s_m_m2io_version\n :::\n 2.0.0\n}\n\n"
    "f_m_ct {\n s_m_title\n r_boltz_ptm\n :::\n \"apo\"\n 0.9\n"
    f" m_atom[{len(_ATOMS)}] {{\n"
    "  i_m_residue_number\n  s_m_chain_name\n  s_m_pdb_residue_name\n"
    "  s_m_pdb_atom_name\n  i_m_atomic_number\n  i_m_formal_charge\n"
    "  r_m_x_coord\n  r_m_y_coord\n  r_m_z_coord\n  r_m_pdb_tfactor\n  :::\n"
    + "".join(
        f'  {i} {r} "{c}" "{rn:<4}" "{an}" {z} 0 {x:.6f} {y:.6f} {w:.6f} 60.00\n'
        for i, (r, c, rn, an, z, x, y, w) in enumerate(_ATOMS, start=1)
    )
    + "  :::\n }\n m_bond[1] {\n  i_m_from\n  i_m_to\n  i_m_order\n  :::\n"
    "  1 1 2 1\n  :::\n }\n}\n"
)


def test_read_residues_and_region_map(tmp_path: Path) -> None:
    structure = tmp_path / "apo.mae"
    structure.write_text(MAE)
    chains = {
        "A": {"sequence": "GA"},
        "B": {"sequence": "A"},
        "C": {"sequence": "AAAA"},  # the binder, a peptide: left out
    }
    residues = read_residues(structure, chains, "C")
    assert [(r.chain, r.resid) for r in residues] == [("A", 1), ("A", 2), ("B", 1)]
    assert residues[0].xyz[0] == 1.0  # glycine: its CA
    assert residues[1].xyz[0] == 7.0  # alanine: its CB

    regions = cluster_regions(residues, 2, pocket_size=1)
    # The results folder does not exist yet, as in a run with --apo.
    csv_path, map_path, pml_path = write_regions(
        tmp_path / "results", structure, regions, chains, "test"
    )
    rows = csv_path.read_text().splitlines()
    assert rows[0] == "chain,residue,residue_name,region,rank,center_distance,pocket"
    assert rows[1].startswith("A,")

    assert map_path.name == "regions.mae"
    text = map_path.read_text()
    atoms = _mae_rows(text, "m_atom")
    assert len(atoms) == len(_ATOMS)
    assert [float(a["r_m_pdb_tfactor"]) for a in atoms] == [1, 1, 1, 1, 1, 2, 2, 0]
    occupancies = [float(a["r_m_pdb_occupancy"]) for a in atoms]
    assert occupancies[5:] == [1.0, 1.0, 0.5]  # B1 is region 2's pocket; LIG is not
    assert sorted(occupancies[:5]) == [0.5, 0.5, 1.0, 1.0, 1.0] or sorted(
        occupancies[:5]
    ) == [0.5, 0.5, 0.5, 1.0, 1.0]  # one of A1, A2 is region 1's pocket
    assert atoms[0]["s_m_pdb_residue_name"] == "GLY "  # the rest is untouched
    assert _mae_rows(text, "m_bond") == [
        {"index": "1", "i_m_from": "1", "i_m_to": "2", "i_m_order": "1"}
    ]
    assert "r_boltz_ptm" in text
    assert pml_path is not None
    assert "load regions.mae, regions" in pml_path.read_text()
