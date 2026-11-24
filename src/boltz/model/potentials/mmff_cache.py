"""MMFF94 ligand terms from RDKit, in the shape ``gaff_cache`` returns.

Ported from BioEmu2's ``denoising/steering/mmff.py``. The term selection, the
unit conversions and the 1-4 / nonbonded split are all kept identical so the
``mmff`` preset steers with the same energy in both forks; the only change is
the entry point, which takes an RDKit molecule (what Boltz has at parse time)
instead of a parameterized ``System``.

Indices are in a unified heavy-plus-hydrogen space: ``AddHs`` appends hydrogens
without renumbering, so a heavy atom keeps its index in the input molecule and
hydrogen ordinals start at ``n_heavy``. Unlike GAFF2 this needs no cache and no
AmberTools -- MMFF typing is RDKit-only and costs milliseconds.
"""

import math

#: mdyn/A to kcal/mol/A^2 (MMFF's bond constant is 143.9325/2 * kb), and
#: MMFF's electrostatic scaling for 1-4 pairs.
_MDYN_TO_KCAL = 143.9325
_ELECTROSTATIC_14 = 0.75


def _properties(mol_w_hs):
    from rdkit.Chem import AllChem

    return AllChem.MMFFGetMoleculeProperties(mol_w_hs)


def _bond_terms(mol_w_hs, properties):
    """Every heavy-atom bond, as [i, j, r0 in A, kb in kcal/mol/A^2].

    GetMMFFBondStretchParams returns (bondType, kb, r0); kb is in mdyn/A, which
    the 143.9325/2 factor converts to kcal/mol/A^2 so this term is in the same
    units as the torsion term's V1/V2/V3. An sp3 C-C is 306.4 and a carbonyl 932.
    """
    out = []
    for bond in mol_w_hs.GetBonds():
        i_atom, j_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        if i_atom.GetAtomicNum() == 1 or j_atom.GetAtomicNum() == 1:
            continue
        params = properties.GetMMFFBondStretchParams(
            mol_w_hs, i_atom.GetIdx(), j_atom.GetIdx()
        )
        if params is None:
            continue
        kb, r0 = float(params[1]), float(params[2])
        out.append(
            [i_atom.GetIdx(), j_atom.GetIdx(), r0, 0.5 * _MDYN_TO_KCAL * kb]
        )
    return out


def _angle_terms(mol_w_hs, properties):
    """Every heavy-atom bond angle, as [i, k, centre, theta0 rad, ka].

    ka is 143.9325 * ka in kcal/mol/rad^2: the cosine well k (1 - cos dtheta)
    has curvature k, so this matches MMFF's harmonic bend.
    """
    out = []
    for center in mol_w_hs.GetAtoms():
        if center.GetAtomicNum() == 1:
            continue
        neighbours = [n for n in center.GetNeighbors() if n.GetAtomicNum() != 1]
        for a in range(len(neighbours)):
            for b in range(a + 1, len(neighbours)):
                i = neighbours[a].GetIdx()
                j = center.GetIdx()
                k = neighbours[b].GetIdx()
                params = properties.GetMMFFAngleBendParams(mol_w_hs, i, j, k)
                if params is None:
                    continue
                # (angleType, ka, theta0 in degrees)
                out.append(
                    [i, k, j, math.radians(float(params[2])),
                     _MDYN_TO_KCAL * float(params[1])]
                )
    return out


def _torsion_terms(mol_w_hs, properties):
    """Every MMFF torsion about a heavy-heavy bond, as [i, j, k, l, V1, V2, V3].

    The central bond j-k must be heavy, but i and l may be hydrogens, so a ring
    C-C bond contributes all nine i-j-k-l paths rather than the single heavy
    one. That matters: an all-atom force field spreads a bond's torsional
    barrier over every path through it, and the heavy path alone is not a weaker
    version of the sum but a different function -- on cyclohexane all nine paths
    minimise at 60 and 180 degrees while the heavy path alone minimises at 180
    and puts a maximum near the chair's 60.

    A torsion is dropped only when RDKit has no parameters for it or when
    V1 = V2 = V3 = 0. There is no scope filter: an earlier version kept only
    ring bonds plus non-ring torsions with V2 > 0, which was a workaround for
    the heavy-atom path having the wrong shape, and reconstructing hydrogens
    fixes that at the source.
    """
    out = []
    for bond in mol_w_hs.GetBonds():
        j_atom, k_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        if j_atom.GetAtomicNum() == 1 or k_atom.GetAtomicNum() == 1:
            continue
        j, k = j_atom.GetIdx(), k_atom.GetIdx()
        for i_atom in j_atom.GetNeighbors():
            if i_atom.GetIdx() == k:
                continue
            for l_atom in k_atom.GetNeighbors():
                if l_atom.GetIdx() == j:
                    continue
                if i_atom.GetIdx() == l_atom.GetIdx():
                    continue  # three-membered ring: i and l are the same atom
                params = properties.GetMMFFTorsionParams(
                    mol_w_hs, i_atom.GetIdx(), j, k, l_atom.GetIdx()
                )
                if params is None:
                    continue
                # (torsionType, V1, V2, V3) -- the leading element is the type
                # index, not V1. Reading params[0:3] puts benzene's
                # planarity-enforcing V2 into the V3 slot and inverts it.
                v1, v2, v3 = (
                    float(params[1]), float(params[2]), float(params[3])
                )
                if v1 == 0.0 and v2 == 0.0 and v3 == 0.0:
                    continue
                out.append(
                    [i_atom.GetIdx(), j, k, l_atom.GetIdx(), v1, v2, v3]
                )
    return out


def _pair_terms(mol_w_hs, properties, *, separation):
    """vdW/electrostatic pairs at a topological separation.

    ``separation=3`` gives the 1-4 pairs, with MMFF's 0.75 electrostatic
    scaling folded into the stored charge product (MMFF applies no scaling to
    the 1-4 van der Waals). ``separation=None`` gives every pair more than three
    bonds apart; the 1-4s are excluded there because they are a separate term
    carrying that scaling, and counting them twice would double them.

    R* is MMFF's pair minimum distance and is kept as such: the potential uses
    MMFF's own buffered 14-7, written in terms of R*, so there is nothing to
    convert.
    """
    from rdkit import Chem

    scale = _ELECTROSTATIC_14 if separation == 3 else 1.0
    dmat = Chem.GetDistanceMatrix(mol_w_hs)
    out = []
    n = mol_w_hs.GetNumAtoms()
    for i in range(n):
        for j in range(i + 1, n):
            d = dmat[i][j]
            if separation == 3:
                if d != 3:
                    continue
            elif d <= 3:
                continue
            params = properties.GetMMFFVdWParams(i, j)
            if params is None:
                continue
            # (R*_unscaled, eps_unscaled, R*, eps) -- the scaled pair is wanted
            r_star, epsilon = float(params[2]), float(params[3])
            q = (
                properties.GetMMFFPartialCharge(i)
                * properties.GetMMFFPartialCharge(j)
                * scale
            )
            out.append([i, j, r_star, epsilon, float(q)])
    return out


def _hydrogen_terms(mol_w_hs, n_heavy):
    """Each hydrogen as {"index", "parent", "r"}, in ordinal order.

    Only the parent and the bond length are stored: the hydrogen is placed from
    its parent's CURRENT heavy neighbours at sampling time, not rebuilt from
    internal coordinates measured on a reference conformer.

    The length is MMFF's own equilibrium X-H, so no conformer is needed at all.
    """
    properties = _properties(mol_w_hs)
    out = []
    for atom in mol_w_hs.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        h = atom.GetIdx()
        neighbours = atom.GetNeighbors()
        if not neighbours:
            continue
        parent = neighbours[0].GetIdx()
        if parent >= n_heavy:
            continue
        r = None
        if properties is not None:
            params = properties.GetMMFFBondStretchParams(mol_w_hs, parent, h)
            if params is not None:
                r = float(params[2])
        if r is None:
            # A hydrogen MMFF cannot type would otherwise be placed at a
            # nonsense distance; the element's ordinary X-H is far better than
            # a guess of zero, and these are rare enough to be worth the
            # fallback rather than dropping the hydrogen and inverting its
            # torsions.
            r = {6: 1.09, 7: 1.01, 8: 0.97, 16: 1.34}.get(
                mol_w_hs.GetAtomWithIdx(parent).GetAtomicNum(), 1.09
            )
        out.append({"index": h, "parent": parent, "r": r})
    return out


def terms_for(mol):
    """MMFF terms for one ligand, or None when RDKit cannot type it.

    ``mol`` is the heavy-atom molecule as Boltz holds it. Returns the same dict
    shape ``gaff_cache.parameters_for`` returns, so the parse step handles both
    force fields identically:

        bonds      [i, j, r0, fc]
        angles     [i, k, centre, theta0, fc]
        torsions   [i, j, k, l, V1, V2, V3]
        pairs      [i, j, rstar, eps, qq]      1-4, 0.75 already applied
        nonbonded  [i, j, rstar, eps, qq]      separation > 3
        hydrogens  [{index, parent, r}, ...]   ordinal order
        n_heavy    the heavy atom count

    Angles come out (outer, outer, CENTRE), which is what ligand_ff's angle
    terms expect.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AddHs
    except Exception:  # noqa: BLE001
        return None

    try:
        heavy = Chem.RemoveHs(Chem.Mol(mol))
    except Exception:  # noqa: BLE001
        return None
    n_heavy = heavy.GetNumAtoms()
    if n_heavy == 0:
        return None

    try:
        # AddHs appends without renumbering, so a heavy atom keeps its index and
        # the hydrogen ordinals start at n_heavy -- the unified space the
        # potentials index into.
        mol_w_hs = AddHs(heavy)
        Chem.SanitizeMol(mol_w_hs)
    except Exception:  # noqa: BLE001
        return None

    properties = _properties(mol_w_hs)
    if properties is None:
        # Not an MMFF-typable molecule (an unsupported element, usually). An
        # optional steering term should add nothing rather than fail a
        # prediction.
        return None

    try:
        return {
            "n_heavy": n_heavy,
            "bonds": _bond_terms(mol_w_hs, properties),
            "angles": _angle_terms(mol_w_hs, properties),
            "torsions": _torsion_terms(mol_w_hs, properties),
            "pairs": _pair_terms(mol_w_hs, properties, separation=3),
            "nonbonded": _pair_terms(mol_w_hs, properties, separation=None),
            "hydrogens": _hydrogen_terms(mol_w_hs, n_heavy),
        }
    except Exception:  # noqa: BLE001
        return None


__all__ = ["terms_for"]
