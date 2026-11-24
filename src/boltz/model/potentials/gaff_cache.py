"""GAFF2 parameters for a SMILES ligand, cached on disk, keyed by atom name.

Deliberately free of any boltz or bioemu2 import; the same file is used by both. Parameterization needs boonza and
AmberTools; sampling needs neither, because it reads this cache. That split is
what lets BioEmu2 run GAFF-steered inside its container, where antechamber is
not installed, after the cache has been warmed once in an environment that has
it:

    python gaff_params.py "CC1CCCCC1" "c1ccccc1"

Run it as a FILE, not with -m: -m imports the package, which pulls in torch,
and the point of this module is to need neither torch nor a GPU.

Atom names, not indices, are the cache key for each term. The molecule is
rebuilt here exactly as the ligand template builder builds it -- AddHs, then
names by RDKit canonical rank -- so a cache entry written in one process maps
onto a batch built in another without depending on either side's atom ordering.

Functional forms are boonza's, following the DMS specification:

    stretch_harm   E = fc (r - r0)^2
    angle_harm     E = fc (theta - theta0)^2, the SECOND atom at the vertex
    dihedral_trig  E = fc0 + sum_n fc_n cos(n phi - phi0), n up to 6

Units are boonza's: angstrom, kcal/mol, degrees, e. Angles are stored here in
degrees as boonza reports them, and converted to radians when packed into
tensors.
"""

from __future__ import annotations

import hashlib
import json
import math
import logging
import os
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# One cache, shared by every consumer: the entries are keyed by canonical
# SMILES and indexed against the canonical molecule, so nothing in them is
# specific to the model that reads them.
CACHE_DIR = Path(
    os.environ.get("GAFF_CACHE", Path.home() / ".cache" / "gaff")
)


def assign_atom_names(mol) -> None:
    """Name ligand atoms by RDKit canonical rank, in place.

    A copy of the ligand template builder's own naming. Duplicated rather than
    imported so that this module needs no package import; the two must stay in
    step, and the unit test checks that they agree.
    """
    from rdkit.Chem.rdmolfiles import CanonicalRankAtoms

    counter: Counter = Counter()
    for can_idx in CanonicalRankAtoms(mol):
        atom = mol.GetAtomWithIdx(can_idx)
        element = atom.GetSymbol().upper()
        counter[element] += 1
        atom.SetProp("name", f"{element}{counter[element]}")


def canonical(smiles: str) -> str | None:
    """RDKit's canonical SMILES, which is the cache key.

    Keying on the canonical form rather than the string given means two
    spellings of one molecule share an entry, and -- more importantly -- a
    consumer holding only a molecule can find the entry by canonicalizing it.
    """
    from rdkit.Chem import MolFromSmiles, MolToSmiles

    mol = MolFromSmiles(smiles)
    return None if mol is None else MolToSmiles(mol)


DEFAULT_PROTEIN_FF = "aa.amber.ff19SB"

#: Bumped whenever the stored FORMAT changes, so an entry written before a new
#: field exists is recomputed instead of being served without it. v2 added the
#: nonbonded parameters and the exclusion list.
# 3: OpenFF entries written under format 2 contain GAFF2 parameters
# (the LIGAND_FF branch was unreachable). Bumping invalidates them
# without deleting anything.
CACHE_FORMAT = 3

# Which ligand force field the terms come from. "gaff2" is the historical
# behaviour; anything starting with "openff" is routed to boonza.openff_patch.
# OpenFF 2.2.1 prefers Cgamma-EXO for 4R-hydroxyproline by 2.48 kcal/mol where
# GAFF2 prefers ENDO by 0.75 -- the truth is 87% exo, so this is the single
# largest correctable error in the benchmark.
# _ligand_patch: both parameterization branches honour this. It was
# read in only one of them until 2026-11-19, which silently gave
# every OpenFF run GAFF2 parameters.
LIGAND_FF: str = "gaff2"


def cache_path(smiles: str,
               protein_ff: str | None = DEFAULT_PROTEIN_FF,
               ligand_ff: str | None = None) -> Path:
    # The force field is part of the key: the same molecule parameterized
    # against ff19SB and ff14SB is not the same set of terms, and ff19SB in
    # particular replaces the backbone phi/psi dihedrals with CMAPs.
    # protein_ff=None is PURE GAFF2 (no protein force field, no pseudopeptide
    # matching). It must key differently, or the two parameterizations
    # overwrite each other's cache entry.
    # LIGAND_FF is part of the key: without it an OpenFF run would read
    # GAFF2 parameters cached by an earlier run and silently steer with the
    # wrong force field.
    key = (f"{canonical(smiles) or smiles}|{protein_ff or 'puregaff2'}"
           f"|{ligand_ff or LIGAND_FF}|v{CACHE_FORMAT}")
    return CACHE_DIR / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"


def parameters_for(smiles: str, *, allow_compute: bool = True,
                   protein_ff: str | None = DEFAULT_PROTEIN_FF,
                   ligand_ff: str | None = None,
                   ) -> dict | None:
    """GAFF2 terms for one SMILES, from the cache or freshly computed.

    With ``allow_compute`` false this only ever reads the cache, which is how
    sampling uses it: a cache miss returns None rather than trying to run
    antechamber in an environment that has no AmberTools.
    """
    lf = ligand_ff or LIGAND_FF
    path = cache_path(smiles, protein_ff, lf)
    if path.exists():
        try:
            data = json.loads(path.read_text())
            # The key already carries the force field; this guards against an
            # entry written under the wrong label, which has happened before.
            if data.get("ligand_ff", lf) == lf:
                return data
            logger.warning("GAFF cache %s holds %s, not %s; recomputing",
                           path, data.get("ligand_ff"), lf)
        except Exception:  # noqa: BLE001
            logger.warning("GAFF cache %s unreadable; recomputing", path)
    if not allow_compute:
        logger.warning(
            "no GAFF parameters cached for %r. Warm the cache where boonza and "
            "AmberTools are installed: python gaff_params.py %r", smiles, smiles)
        return None
    try:
        return compute(smiles, protein_ff=protein_ff, ligand_ff=lf)
    except ImportError:
        # An optional restraint must not kill a prediction. Without boonza
        # there is nothing to compute from, so warn and contribute no terms --
        # raising here took down three ring runs over two uncached ligands.
        logger.warning(
            "no GAFF parameters cached for %r and boonza is not importable "
            "here, so --gaff contributes nothing for this ligand. Warm the "
            "cache where boonza and AmberTools are installed: "
            "python gaff_params.py %r", smiles, smiles)
        return None


def compute(smiles: str, protein_ff: str | None = DEFAULT_PROTEIN_FF,
            charge_geometry: str = "reference",
            ligand_ff: str | None = None) -> dict | None:
    """Parameterize with GAFF2 through boonza and write the cache entry."""
    # The ligand force field, e.g. "gaff2" or "openff-2.2.1"; None is LIGAND_FF.
    lf = ligand_ff or LIGAND_FF
    try:
        import boonza
    except ImportError as exc:
        raise ImportError(
            "GAFF steering needs boonza (github.com/siyoungkimlab/boonza) and "
            "AmberTools for AM1-BCC charges. The MMFF terms need neither."
        ) from exc

    from rdkit.Chem import AddHs, MolFromSmiles

    mol = MolFromSmiles(smiles)
    if mol is None:
        logger.warning("RDKit cannot parse %r", smiles)
        return None

    # boonza.from_smiles does the 3D work properly: AddHs, several ETKDG
    # conformers from random coordinates, MMFF94 (UFF where MMFF has no
    # parameters), and it keeps the lowest-energy one. It also warns when the
    # SMILES leaves a stereocentre open. AM1-BCC runs sqm on these coordinates,
    # and an unembedded molecule -- every atom at the origin -- makes antechamber
    # fail with "unable to find sqm charges".
    system = boonza.from_smiles(smiles, name="LIG")

    # Names come from a molecule built the same way the ligand template builder
    # builds it, matched to the system POSITIONALLY: from_smiles passes
    # AddHs(MolFromSmiles(smiles)) straight to from_rdkit, so atom order is
    # preserved (verified for a ring and a chiral peptide). boonza's own names
    # follow a different order, so they are not used.
    mol_w_hs = AddHs(mol)
    heavy = {i for i, a in enumerate(mol_w_hs.GetAtoms()) if a.GetAtomicNum() != 1}
    if system.natoms != mol_w_hs.GetNumAtoms():
        logger.warning("boonza returned %d atoms for %d; skipping %r",
                       system.natoms, mol_w_hs.GetNumAtoms(), smiles)
        return None
    if list(system.atoms["anum"]) != [a.GetAtomicNum() for a in mol_w_hs.GetAtoms()]:
        logger.warning("atom order differs from the source molecule for %r; "
                       "skipping rather than mapping terms wrongly", smiles)
        return None

    # Terms are stored as indices into the CANONICAL molecule's atom order, so
    # a consumer with its own molecule maps them by substructure match and
    # never has to share an atom-naming scheme. AddHs appends, so heavy indices
    # here are already MolFromSmiles order; the canonical order may differ, and
    # this match carries terms across.
    canon_smiles = canonical(smiles)
    canon = MolFromSmiles(canon_smiles)
    match = mol.GetSubstructMatch(canon)          # canon index -> mol index
    if len(match) != canon.GetNumAtoms():
        logger.warning("cannot match %r onto its own canonical form; skipping",
                       smiles)
        return None
    to_canon = {mol_idx: canon_idx for canon_idx, mol_idx in enumerate(match)}

    # pseudopeptide="protein" finds the amino acids inside a ligand and gives
    # them the protein force field. It only does anything when one is supplied:
    # gaff2_patch with no force fields builds a complete GAFF2 field, and a
    # peptide ligand then comes back with no CMAP at all.
    if protein_ff is None:
        # Pure GAFF2: gaff2_patch with NO force fields builds a complete GAFF2
        # field, so there is nothing to merge and no pseudopeptide matching.
        # A CMAP cannot appear here -- it is a protein force field term.
        if lf.startswith("openff"):
            # Same return type as gaff2_patch, so the extraction below is
            # unchanged. charge_model="am1bcc" because openff-2.3.0 defaults
            # to NAGL, which needs a torch backend that is not installed --
            # and fixing the charges also isolates the force-field change.
            merged = boonza.openff_patch(
                system, [], ligandff=lf, charge_model="am1bcc",
                charge_geometry=charge_geometry)
        else:
            merged = boonza.gaff2_patch(
                system, [], charge_geometry=charge_geometry)
    else:
        # LIGAND_FF must be honoured here too. This branch used to call
        # gaff2_patch unconditionally while the entry was still stamped with
        # LIGAND_FF, so every OpenFF run silently produced GAFF2 parameters
        # under an OpenFF label -- and compute() defaults protein_ff to
        # DEFAULT_PROTEIN_FF, so this was the branch that always ran.
        if lf.startswith("openff"):
            # openff_patch takes the same (system, forcefields) positional
            # pair. It has no pseudopeptide argument, so a PEPTIDE ligand gets
            # no CMAP and no protein types for its residues under OpenFF; for
            # the small molecules this is used on there is nothing to match.
            patch = boonza.openff_patch(
                system, [protein_ff], ligandff=lf,
                charge_model="am1bcc", charge_geometry=charge_geometry)
        else:
            patch = boonza.gaff2_patch(system, [protein_ff],
                                       pseudopeptide="protein",
                                       charge_geometry=charge_geometry)
        merged = boonza.merge_forcefields(protein_ff, patch)
    system = boonza.parameterize(system, [merged])

    def rows(table: str):
        if table in system.table_names:
            handle = system.table(table)
            for i in range(handle.nterms):
                yield handle.term(i)

    data: dict = {"smiles": smiles, "canonical_smiles": canon_smiles,
                  "protein_ff": protein_ff,
                  "ligand_ff": lf,
                  "bonds": [], "angles": [], "torsions": [], "cmaps": [],
                  "pairs": [], "nonbonded": [], "exclusions": [],
                  "hydrogens": [], "n_heavy": None}

    for term in rows("stretch_harm"):
        i, j = term.atom_ids
        if i in heavy and j in heavy:
            data["bonds"].append([to_canon[i], to_canon[j], term["r0"], term["fc"]])

    for term in rows("angle_harm"):
        i, j, k = term.atom_ids              # vertex is the SECOND atom
        if {i, j, k} <= heavy:
            data["angles"].append([to_canon[i], to_canon[k], to_canon[j],
                                   term["theta0"], term["fc"]])

    # Hydrogens, as internal coordinates against three heavy reference atoms.
    #
    # Dropping every term that touches a hydrogen does not merely weaken the
    # torsion potential, it inverts it: GAFF spreads a bond's barrier over all
    # nine i-j-k-l paths, and the single heavy C-C-C-C path turns the chair's
    # +-55 degree minimum into a maximum (measured: all nine paths give minima
    # at 60 and 180 degrees, the heavy path alone only at 180). So the
    # hydrogens are kept and reconstructed at sampling time.
    #
    # Each is stored as (r, theta, phi) against (C, B, A) -- its parent and two
    # heavy atoms further out. Only the parent and r are used now: the steering
    # places every hydrogen from its parent's current heavy neighbours
    # (potentials._place_hydrogens), because phi measured on one reference
    # conformer is wrong for any other ring shape or chain torsion.
    import numpy as _np

    pos = _np.array(system.atoms["pos"])

    def _dihedral(a, b, c, d):
        b0, b1, b2 = pos[a] - pos[b], pos[c] - pos[b], pos[d] - pos[c]
        b1 = b1 / _np.linalg.norm(b1)
        v = b0 - b0.dot(b1) * b1
        w = b2 - b2.dot(b1) * b1
        return math.degrees(math.atan2(_np.cross(b1, v).dot(w), v.dot(w)))

    def _angle(a, b, c):
        u, v = pos[a] - pos[b], pos[c] - pos[b]
        u = u / _np.linalg.norm(u)
        v = v / _np.linalg.norm(v)
        return math.degrees(math.acos(max(-1.0, min(1.0, float(u.dot(v))))))

    hydrogens: dict[int, int] = {}        # mol index -> index in the unified space
    next_index = len(to_canon)
    for atom in mol_w_hs.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        h = atom.GetIdx()
        parent = atom.GetNeighbors()[0].GetIdx()
        if parent not in heavy:
            continue
        # B: a heavy neighbour of the parent. A: one step further out, falling
        # back to another neighbour of the parent for a two-heavy-atom molecule.
        b_candidates = [n.GetIdx() for n in mol_w_hs.GetAtomWithIdx(parent).GetNeighbors()
                        if n.GetIdx() in heavy]
        if not b_candidates:
            continue
        b = b_candidates[0]
        a_candidates = [n.GetIdx() for n in mol_w_hs.GetAtomWithIdx(b).GetNeighbors()
                        if n.GetIdx() in heavy and n.GetIdx() != parent]
        a_candidates += [x for x in b_candidates if x != b]
        if not a_candidates:
            continue
        a = a_candidates[0]
        hydrogens[h] = next_index
        data.setdefault("hydrogens", []).append({
            "index": next_index,
            "parent": to_canon[parent],
            "b": to_canon[b],
            "a": to_canon[a],
            "r": float(_np.linalg.norm(pos[h] - pos[parent])),
            "theta": _angle(h, parent, b),
            "phi": _dihedral(h, parent, b, a),
        })
        next_index += 1

    def _unified(i: int):
        """Index in the heavy-plus-hydrogen space, or None if unplaceable."""
        if i in heavy:
            return to_canon[i]
        return hydrogens.get(i)

    aromatic_bonds = {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
        for b in mol_w_hs.GetBonds() if b.GetIsAromatic()
    }

    for term in rows("dihedral_trig"):
        raw = [int(x) for x in term.atom_ids]
        # Every torsion is kept, aromatic bonds included. They are the
        # stiffest terms and they do set the stability ceiling, so the weight
        # has to respect it rather than the term being dropped.
        ids = [_unified(x) for x in raw]
        if any(x is None for x in ids):
            continue
        fcs = [term[f"fc{n}"] for n in range(7)]
        if not any(fcs[1:]):                 # fc0 alone is a constant offset
            continue
        data["torsions"].append([ids[0], ids[1], ids[2], ids[3],
                                 term["phi0"], fcs])

    # CMAP: a 2D surface over a coupled torsion pair, which for a peptide-like
    # ligand is the backbone phi/psi. ff19SB carries these in place of the
    # backbone dihedrals, so they are the term that shapes a Ramachandran plot.
    # The grid is stored with the entry rather than referenced, so a consumer
    # needs nothing but this file.
    for term in rows("torsiontorsion_cmap"):
        ids = term.atom_ids
        if not set(ids) <= heavy:
            continue
        mapped = [to_canon.get(a) for a in ids]
        if any(m is None for m in mapped):
            continue
        grid = system.aux_tables[term["cmapid"]]
        size = int(round(len(grid) ** 0.5))
        if size * size != len(grid):
            logger.warning("CMAP grid for %r is not square (%d points); skipping",
                           smiles, len(grid))
            continue
        # Sorted by (phi, psi) so the consumer can reshape without re-sorting.
        order = sorted(range(len(grid)),
                       key=lambda i: (float(grid["phi"][i]), float(grid["psi"][i])))
        data["cmaps"].append({
            "atom_idxs": mapped,
            "size": size,
            "phi_min": float(min(grid["phi"])),
            "psi_min": float(min(grid["psi"])),
            "energy": [float(grid["energy"][i]) for i in order],
        })

    # 1-4 pairs, verbatim from pair_12_6_es: the scaling the force field
    # applies (SCNB/SCEE) is already in aij, bij and qij, so rescaling here
    # would apply it twice. A/r^12 - B/r^6 + 332.0637 q_i q_j / r.
    for term in rows("pair_12_6_es"):
        ids = [_unified(int(x)) for x in term.atom_ids]
        if any(x is None for x in ids):
            continue
        data["pairs"].append([ids[0], ids[1], term["aij"], term["bij"], term["qij"]])

    # Per-atom nonbonded parameters, for the full nonbonded term. sigma
    # and epsilon come from the nonbonded table and the charge from the atom.
    for term in rows("nonbonded"):
        ids = [_unified(int(x)) for x in term.atom_ids]
        if any(x is None for x in ids):
            continue
        atom = system.atoms[int(term.atom_ids[0])]
        data["nonbonded"].append([ids[0], term["sigma"], term["epsilon"],
                                  float(atom.charge)])

    # Pairs NOT to compute: 1-2 and 1-3 bonded neighbours, and the 1-4 pairs,
    # which are a separate term with the force field's own scaling applied.
    # Counting them here as well would double them.
    for term in rows("exclusion"):
        ids = [_unified(int(x)) for x in term.atom_ids]
        if any(x is None for x in ids):
            continue
        data["exclusions"].append([ids[0], ids[1]])

    data["n_heavy"] = len(to_canon)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(smiles, protein_ff, lf)
    # One tmp file per writer. Keyed on the molecule hash alone, concurrent
    # tasks parameterizing the same molecule shared this path, and the first
    # rename left the rest with a FileNotFoundError -- 4 of 20 experiments in
    # one optimization round. The rename itself stays atomic, so a reader
    # still never sees a half-written entry.
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)   # a crash must not leave scratch behind
    logger.info("cached GAFF parameters for %r at %s", smiles, path)
    return data


def map_to(mol, data: dict) -> dict[int, int] | None:
    """{canonical atom index -> index in ``mol``} for a cached entry.

    ``mol`` is the consumer's own heavy-atom molecule. The correspondence is a
    substructure match against the canonical molecule the terms were indexed
    against, so neither side needs the other's atom ordering or naming scheme.
    Returns None when the molecule does not match the entry, which is the right
    answer for a cache hit on a different protonation state or tautomer.
    """
    from rdkit.Chem import MolFromSmiles, RemoveHs

    canon = MolFromSmiles(data["canonical_smiles"])
    if canon is None:
        return None
    try:
        heavy_mol = RemoveHs(mol)
    except Exception:  # noqa: BLE001
        heavy_mol = mol
    match = heavy_mol.GetSubstructMatch(canon)
    if len(match) != canon.GetNumAtoms():
        return None
    return {canon_idx: mol_idx for canon_idx, mol_idx in enumerate(match)}


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not argv:
        print(__doc__)
        return 2
    failures = 0
    for smiles in argv:
        data = compute(smiles)
        if data is None:
            failures += 1
            print(f"  FAILED  {smiles}")
            continue
        print(f"  ok  {smiles}: {len(data['bonds'])} bonds, "
              f"{len(data['angles'])} angles, {len(data['torsions'])} torsions, "
              f"{len(data.get('pairs', []))} 1-4 pairs, "
              f"{len(data.get('exclusions', []))} exclusions, "
              f"{len(data.get('cmaps', []))} cmaps -> {cache_path(smiles).name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
