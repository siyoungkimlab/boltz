"""Ligand force-field steering terms: GAFF2, OpenFF and MMFF94.

Ported from BioEmu2's ``denoising/steering/potentials.py``, which is the
reference implementation these forms were benchmarked in. The functional forms,
core switches, hydrogen placement and per-term weights are kept identical so a
Boltz run and a BioEmu2 run of the same preset steer with the same energy; the
only changes are to fit Boltz's ``Potential`` API (``compute(coords, feats,
parameters)`` plus ``compute_gradient``) and its feature dictionary.

Three presets are exposed through ``get_ligand_potentials``:

    gaff2, openff-2.2.1   GAFF2 / OpenFF bond, angle, 1-4, torsion, nonbonded
    mmff                  the MMFF94 counterparts

Each is the physicality set MINUS the three terms that override a ligand force
field (PoseBusters, bond length, bond angle), plus the five force-field terms at
the weights in ``LIGAND_STEER_*``.
"""

import logging
import math
from typing import Optional

import torch

from boltz.model.potentials.potentials import (
    Potential,
    get_potentials,
)
from boltz.model.potentials.schedules import (
    LinearInterpolation,
    PiecewiseStepFunction,
)

logger = logging.getLogger(__name__)

#: Coulomb constants, kcal.A/mol/e^2.
_MMFF_COULOMB = 332.0716
_GAFF_COULOMB = 332.0637

#: Smallest core switch, angstrom.
_CORE_FLOOR = 0.5


# --------------------------------------------------------------------------
# Bounded functional forms
# --------------------------------------------------------------------------


def _connected_zero(coords: torch.Tensor) -> torch.Tensor:
    """Per-particle zero with a differentiable path to coordinates."""
    return coords[..., :0, :].sum(dim=(-2, -1))


def _core_switch(size: torch.Tensor, frac: float = 0.5) -> torch.Tensor:
    """s = max(frac * size, 0.5 A), size being the pair's LJ sigma (GAFF) or R* (MMFF).

    For the Coulomb core frac is the term's coulomb_core_switch (0.5 by default
    everywhere, so short hydrogen-bond contacts keep real Coulomb). For the vdW
    core it is the term's lj_core_switch: 0.7 for GAFF2/OpenFF's 12-6 (at
    sigma/2 the r^-13 wall already pushes ~6000 kcal/mol/A on a C...C pair, and
    at guidance weight 0.005 that kicks atoms ~30 A per step against a bond
    pull-back of ~0.1-0.3 A, so compressed early predictions exploded), 0.5 for
    MMFF's buffered 14-7 (R*/2, whose wall is already ~20x softer there).
    """
    return torch.clamp(frac * size, min=_CORE_FLOOR)


def _parabola_core(r, s, e_s, de_s, outer):
    """``outer`` for r >= s; inside, E = A - B r^2 with the value e_s and slope
    de_s of the outer form at s, so energy and force are continuous and
    E(0) = A is finite. ``outer`` must already be evaluated at max(r, s), never
    below s, so no masked branch carries an infinite gradient."""
    b = -de_s / (2.0 * s)
    a = e_s + b * s * s
    return torch.where(r < s, a - b * r * r, outer)


def _coulomb_core(r, k, s):
    """Real Coulomb k / r for r >= s, with k = C q_i q_j / D (scaling included);
    a parabola matching value and slope inside: (k / 2s)(3 - r^2 / s^2)."""
    rr = torch.maximum(r, s)
    return _parabola_core(r, s, k / s, -k / (s * s), k / rr)


def _lj126_core(r, a, b, s):
    """a / r^12 - b / r^6 for r >= s, parabola inside (see _parabola_core)."""
    rr = torch.maximum(r, s)
    s6 = s**6
    e_s = a / (s6 * s6) - b / s6
    de_s = -12.0 * a / (s6 * s6 * s) + 6.0 * b / (s6 * s)
    r6 = rr**6
    return _parabola_core(r, s, e_s, de_s, a / (r6 * r6) - b / r6)


def _mmff147(r, rstar, eps):
    """MMFF's buffered 14-7 and its derivative in r."""
    rs7 = rstar**7
    u = r + 0.07 * rstar
    f = (1.07 * rstar / u) ** 7
    w = r**7 + 0.12 * rs7
    g = 1.12 * rs7 / w - 2.0
    df = -7.0 * f / u
    dg = -1.12 * rs7 * 7.0 * r**6 / (w * w)
    return eps * f * g, eps * (df * g + f * dg)


def _mmff147_core(r, rstar, eps, s):
    """MMFF's buffered 14-7 for r >= s, parabola inside (see _parabola_core)."""
    e_s, de_s = _mmff147(s, rstar, eps)
    outer, _ = _mmff147(torch.maximum(r, s), rstar, eps)
    return _parabola_core(r, s, e_s, de_s, outer)


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(1e-9)


def _angle_value(coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Bond angle at index[2], via atan2 rather than acos: acos' is unbounded at
    a straight angle, where a nitrile or an alkyne sits exactly."""
    u = coords.index_select(-2, index[0]) - coords.index_select(-2, index[2])
    v = coords.index_select(-2, index[1]) - coords.index_select(-2, index[2])
    u = _unit(u)
    v = _unit(v)
    cosine = (u * v).sum(dim=-1)
    sine = torch.linalg.norm(torch.cross(u, v, dim=-1), dim=-1)
    return torch.atan2(sine, cosine)


def _dihedral_value(coords: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Signed dihedral for each column of index [4, N], radians."""
    p0 = coords.index_select(-2, index[0])
    p1 = coords.index_select(-2, index[1])
    p2 = coords.index_select(-2, index[2])
    p3 = coords.index_select(-2, index[3])
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = _unit(b1)
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    x = (v * w).sum(-1)
    y = (torch.cross(b1, v, dim=-1) * w).sum(-1)
    return torch.atan2(y, x)


# --------------------------------------------------------------------------
# Hydrogen placement
# --------------------------------------------------------------------------

#: Ideal angles for hydrogens placed from neighbours, radians. BENT is the X-P-H
#: angle of a two-coordinate parent (hydroxyl, thiol), by atomic number.
_H_TET = math.radians(109.4712)
_H_TRI = math.radians(120.0)
_H_BENT = {8: math.radians(108.5), 16: math.radians(96.0)}
#: Rotations tried for a rotatable hydrogen (OH, SH), as the dihedral H-P-B-D in
#: degrees, by the coordination of B (the atom the O or S is bonded to,
#: hydrogens included). On a trigonal B (phenol, enol) the minima are in the
#: plane; on a tetrahedral one (an alcohol) they are staggered.
_H_ROTAMERS_BY_COORD = {3: (0.0, 180.0), 4: (60.0, 180.0, 300.0)}
_H_ROTAMERS = tuple(float(d) for d in range(0, 360, 30))

#: Coordinates with hydrogens, set while a _HydrogenGroup scores rotamers.
_HYDROGEN_OVERRIDE: Optional[torch.Tensor] = None


def _hydrogen_plan(parent, bonds, element):
    """Group hydrogens by the rule that places them, from topology alone.

    ``parent[k]`` is hydrogen k's heavy parent, ``bonds`` the heavy-heavy bonds
    and ``element`` the atomic number of every heavy atom. Coordination is heavy
    neighbours plus hydrogens:

      3 heavy + 1 H   (CH, NH+)          tet3: opposite the sum of bond directions
      2 heavy + 2 H   (CH2)              tet2: +-54.7 deg out of the X-P-X plane
      2 heavy + 1 H   (=CH-, ring NH)    tri2: in plane, opposite the bisector
      1 heavy, 4-coordinate (CH3, NH3+)  term: staggered, dihedral 180 / +60 / -60
      1 heavy, 3-coordinate (NH2, =CH2)  term: planar, dihedral 0 / 180
      1 heavy, 2-coordinate sp C (#CH)   linear: along the B->P axis
      1 heavy, 2-coordinate (OH, SH)     rotor: lowest-energy rotamer

    Dihedrals are H-P-B-D with B the parent's heavy neighbour and D a heavy
    neighbour of B. A hydrogen no rule covers raises: there is no fallback,
    because a silently misplaced hydrogen changes every torsion through it.
    """
    nb: dict = {}
    for i, j in bonds:
        nb.setdefault(i, set()).add(j)
        nb.setdefault(j, set()).add(i)
    hs: dict = {}
    for k, p in enumerate(parent):
        hs.setdefault(p, []).append(k)
    plan: dict = {}
    uncovered: list = []
    for p, ks in hs.items():
        heavy = sorted(nb.get(p, ()))
        coord = len(heavy) + len(ks)
        if len(heavy) == 3 and len(ks) == 1:
            plan.setdefault("tet3", []).append((ks[0], p, *heavy))
        elif len(heavy) == 2 and len(ks) == 2 and coord == 4:
            plan.setdefault("tet2", []).append((ks[0], ks[1], p, *heavy))
        elif len(heavy) == 2 and len(ks) == 1:
            plan.setdefault("tri2", []).append((ks[0], p, *heavy))
        elif len(heavy) == 1:
            b = heavy[0]
            d = sorted(x for x in nb.get(b, ()) if x != p)
            if coord == 2 and len(ks) == 1 and element[p] == 6:
                # sp carbon: one heavy neighbour plus one hydrogen can only be
                # a terminal alkyne, whose C-H lies on the extension of the
                # B->P axis. It needs its own rule rather than an _H_BENT
                # entry of 180 deg, because at 180 the bend-and-dihedral
                # construction is degenerate -- e1 is the unit part of a
                # vector that has become parallel to a -- and would give NaN.
                # No dihedral is needed, so this is placed before the `d`
                # guard and covers acetylene itself, where B has no other
                # heavy neighbour. Carbon only: a 2-coordinate N-H (an imine)
                # is bent, not linear, and divalent carbenes are not in scope.
                plan.setdefault("linear", []).append((ks[0], p, b))
            elif not d:
                uncovered += ks
            elif coord == 4 and len(ks) == 3:
                for k, phi in zip(ks, (180.0, 60.0, -60.0)):
                    plan.setdefault("term", []).append(
                        (k, p, b, d[0], _H_TET, math.radians(phi))
                    )
            elif coord == 3 and len(ks) == 2:
                for k, phi in zip(ks, (0.0, 180.0)):
                    plan.setdefault("term", []).append(
                        (k, p, b, d[0], _H_TRI, math.radians(phi))
                    )
            elif coord == 2 and len(ks) == 1 and element[p] in _H_BENT:
                b_coord = len(nb.get(b, ())) + len(hs.get(b, ()))
                cands = _H_ROTAMERS_BY_COORD.get(b_coord, _H_ROTAMERS)
                plan.setdefault("rotor", []).append(
                    (ks[0], p, b, d[0], _H_BENT[element[p]], cands)
                )
            else:
                uncovered += ks
        else:
            uncovered += ks
    if uncovered:
        raise NotImplementedError(
            f"no placement rule for hydrogens {sorted(uncovered)} (parents "
            f"{sorted({parent[k] for k in uncovered})})"
        )
    return plan


def _dihedral_hydrogens(coords, p, b, d, r, theta, phi):
    """Hydrogens at angle theta from the P->B bond and dihedral phi against D:

        P + r [a cos(theta) + sin(theta) (cos(phi) e1 + sin(phi) e2)]

    a = unit(B - P), e1 the unit part of (D - B) perpendicular to a, e2 = a x e1.
    """

    def at(i):
        return coords.index_select(-2, torch.as_tensor(i, device=coords.device))

    p0, b0, d0 = at(p), at(b), at(d)
    a = _unit(b0 - p0)
    w = d0 - b0
    e1 = _unit(w - (w * a).sum(-1, keepdim=True) * a)
    e2 = torch.cross(a, e1, dim=-1)
    theta, phi = theta.unsqueeze(-1), phi.unsqueeze(-1)
    return p0 + r.unsqueeze(-1) * (
        a * torch.cos(theta)
        + torch.sin(theta) * (torch.cos(phi) * e1 + torch.sin(phi) * e2)
    )


def _place_hydrogens(coords, plan, r, rotor_phi=None):
    """Append hydrogens to ``coords``, each built from its parent's CURRENT heavy
    neighbours (see ``_hydrogen_plan``). Only the bond length ``r`` [H] comes
    from the parameterization; no reference conformer is used.

    ``rotor_phi`` [..., n_rotors] sets the dihedral of each rotatable hydrogen,
    radians; anti (pi) when None. Differentiable in ``coords`` throughout.

    This deliberately replaces a NeRF rebuild from (r, theta, phi) measured on a
    reference conformer. phi is conformational: a ring CH2 hydrogen fixed
    against one ring neighbour ignores the other, so for cyclohexane, whose
    reference was a twist-boat, hydrogens on GAFF2 MD chairs came out 0.5 A from
    MD's and down to 0.6 A from a ring carbon, and the steering energy put the
    chair 62 kcal/mol above the twist-boat. Placed from neighbours they are
    0.10 A from MD's.
    """
    n_h = int(r.shape[-1])
    dev, dt = coords.device, coords.dtype
    out = torch.cat(
        (coords, coords.new_zeros((*coords.shape[:-2], n_h, 3))), dim=-2
    )
    if n_h == 0:
        return out
    r = r.to(dt)
    idx: list = []
    pos: list = []

    def at(i):
        return coords.index_select(-2, torch.as_tensor(i, device=dev))

    if plan.get("tet3"):
        k, p, a, b, c = map(list, zip(*plan["tet3"]))
        p0 = at(p)
        s = _unit(at(a) - p0) + _unit(at(b) - p0) + _unit(at(c) - p0)
        idx += k
        pos.append(p0 - r[k].unsqueeze(-1) * _unit(s))
    if plan.get("linear"):
        k, p, b = map(list, zip(*plan["linear"]))
        p0 = at(p)
        idx += k
        pos.append(p0 - r[k].unsqueeze(-1) * _unit(at(b) - p0))
    if plan.get("tet2"):
        k0, k1, p, a, b = map(list, zip(*plan["tet2"]))
        p0 = at(p)
        u, v = _unit(at(a) - p0), _unit(at(b) - p0)
        bis, nrm = _unit(u + v), _unit(torch.cross(u, v, dim=-1))
        cos_h, sin_h = math.cos(_H_TET / 2), math.sin(_H_TET / 2)
        idx += k0 + k1
        pos.append(p0 + r[k0].unsqueeze(-1) * (-bis * cos_h + nrm * sin_h))
        pos.append(p0 + r[k1].unsqueeze(-1) * (-bis * cos_h - nrm * sin_h))
    if plan.get("tri2"):
        k, p, a, b = map(list, zip(*plan["tri2"]))
        p0 = at(p)
        bis = _unit(_unit(at(a) - p0) + _unit(at(b) - p0))
        idx += k
        pos.append(p0 - r[k].unsqueeze(-1) * bis)
    if plan.get("term"):
        k, p, b, d, th, ph = map(list, zip(*plan["term"]))
        idx += k
        pos.append(
            _dihedral_hydrogens(
                coords,
                p,
                b,
                d,
                r[k],
                torch.tensor(th, device=dev, dtype=dt),
                torch.tensor(ph, device=dev, dtype=dt),
            )
        )
    if plan.get("rotor"):
        k, p, b, d, th = (list(x) for x in list(zip(*plan["rotor"]))[:5])
        if rotor_phi is None:
            rotor_phi = torch.full(
                (*coords.shape[:-2], len(k)), math.pi, device=dev, dtype=dt
            )
        idx += k
        pos.append(
            _dihedral_hydrogens(
                coords,
                p,
                b,
                d,
                r[k],
                torch.tensor(th, device=dev, dtype=dt),
                rotor_phi.to(dt),
            )
        )
    pos_t = torch.cat(pos, dim=-2).to(out.dtype)
    n = coords.shape[-2]
    return out.index_copy(-2, torch.as_tensor(idx, device=dev) + n, pos_t)


class _HydrogenGroup:
    """Potentials that see the same hydrogens, so a rotatable hydrogen is placed
    at the rotamer of lowest summed energy over all of them rather than each
    term choosing its own. The factories share one group across their terms;
    a potential built on its own forms a group by itself.
    """

    def __init__(self, members=()):
        self.members = list(members)
        self._choice = None  # (fingerprint, rotor dihedrals)

    @staticmethod
    def _fingerprint(c: torch.Tensor) -> tuple:
        c = c.detach().double()
        w = torch.arange(
            1, c.numel() + 1, device=c.device, dtype=c.dtype
        ).reshape(c.shape)
        return (
            tuple(c.shape),
            str(c.device),
            round(float(c.sum()), 6),
            round(float((c * w).sum()), 3),
        )

    def rotamers(self, coords, feats, plan, r):
        """[..., n_rotors] dihedrals of lowest summed energy, one rotor at a time
        with the others held (anti to start), each from its own candidates
        (_H_ROTAMERS_BY_COORD). Chosen without gradients."""
        global _HYDROGEN_OVERRIDE
        fp = self._fingerprint(coords)
        if self._choice is not None and self._choice[0] == fp:
            return self._choice[1]
        c = coords.detach()
        n_rot = len(plan["rotor"])
        phis = torch.full(
            (*c.shape[:-2], n_rot), math.pi, device=c.device, dtype=c.dtype
        )
        with torch.no_grad():
            for j in range(n_rot):
                cand = torch.deg2rad(
                    torch.tensor(
                        plan["rotor"][j][5], device=c.device, dtype=c.dtype
                    )
                )
                energies = []
                for phi in cand:
                    trial = phis.clone()
                    trial[..., j] = phi
                    _HYDROGEN_OVERRIDE = _place_hydrogens(c, plan, r, trial)
                    try:
                        energies.append(
                            sum(m.compute(c, feats, None) for m in self.members)
                        )
                    finally:
                        _HYDROGEN_OVERRIDE = None
                phis[..., j] = cand[torch.stack(energies, -1).argmin(-1)]
        self._choice = (fp, phis)
        return phis


def _share_hydrogen_group(potentials) -> None:
    """One _HydrogenGroup for every hydrogen-using term of a force field."""
    members = [
        p
        for p in potentials
        if isinstance(p, HydrogenMixin) and getattr(p, "use_hydrogens", False)
    ]
    group = _HydrogenGroup(members)
    for p in members:
        p._hydrogen_group = group


class AutogradGradientMixin:
    """compute_gradient by autograd instead of a hand-written derivative.

    Worth it where the analytic form is error-prone rather than merely tedious:
    a wrong derivative is silent, since the energy still looks right and only
    the forces are wrong. This is also exactly how BioEmu2 evaluates these
    terms, so the two implementations take the same gradient path.

    These terms build their own energy in ``compute`` rather than going through
    Boltz's ``compute_args``/``compute_variable``/``compute_function`` split,
    which assumes one variable per term and cannot express a pair term whose
    size enters the core switch. The three abstract hooks are therefore
    satisfied with raisers: reaching one means ``compute`` was bypassed.
    """

    def compute_function(self, value, *args, negation_mask=None,
                         compute_derivative=False):
        raise NotImplementedError(
            f"{type(self).__name__} computes its energy in compute()"
        )

    def compute_variable(self, coords, index, compute_gradient=False):
        raise NotImplementedError(
            f"{type(self).__name__} computes its energy in compute()"
        )

    def compute_args(self, t, feats, **parameters):
        raise NotImplementedError(
            f"{type(self).__name__} computes its energy in compute()"
        )

    def compute_gradient(self, coords, feats, parameters):
        # Lightning runs predict_step under torch.inference_mode(), where
        # autograd cannot record and torch.enable_grad() does NOT lift it:
        # the energy comes back without requires_grad and every gradient is
        # silently zero. Cloning inside inference_mode(False) yields a normal
        # tensor that autograd can track. Upstream Boltz's own potentials use
        # analytic derivatives, so only these autograd terms were affected.
        with torch.inference_mode(False), torch.enable_grad():
            work = coords.clone().detach().requires_grad_(True)
            energy = self.compute(work, feats, parameters)
            if not getattr(energy, "requires_grad", False):
                msg = (f"{type(self).__name__}: energy has no grad_fn, so the "
                       "gradient would be zero. This silently disabled all "
                       "ligand force-field guidance once already.")
                raise RuntimeError(msg)
            (grad,) = torch.autograd.grad(energy.sum(), work)
        return grad.clone()


class HydrogenMixin:
    """Builds ligand hydrogens before evaluating, when asked to.

    The models predict heavy atoms only, but an all-atom force field spreads a
    bond's torsional barrier over every i-j-k-l path through it. Keeping only
    the heavy path does not weaken the term, it inverts it: measured on
    cyclohexane, all nine paths minimise at 60 and 180 degrees while the heavy
    path alone minimises at 180 and puts a MAXIMUM at the chair's 60.

    Hydrogens are placed from each parent's current heavy neighbours
    (_place_hydrogens), with rotatable ones (OH, SH) at the rotamer of lowest
    energy summed over the potentials sharing this hydrogen group.

    With ``use_hydrogens`` false the terms naming a hydrogen are skipped and the
    heavy-only behaviour is recovered, which is what the two are compared
    against.
    """

    _H_REF_KEY = ""
    _H_GEOM_KEY = ""
    _H_PARENT_KEY = ""
    _H_BOND_KEY = ""

    def _hydrogen_plan_for(self, feats):
        ref = feats[self._H_REF_KEY][0]
        cache = self.__dict__.setdefault("_hydrogen_plans", {})
        key = (int(ref.shape[-1]), int(feats["atom_pad_mask"].shape[1]))
        if key not in cache:
            n = key[1]  # hydrogens are indexed from the padded width
            bonds = feats[self._H_BOND_KEY][0].cpu()
            bonds = bonds[:, (bonds < n).all(0)]
            element = feats["ref_element"][0].argmax(-1).cpu().tolist()
            cache[key] = _hydrogen_plan(
                [int(x) for x in ref[0].cpu().tolist()],
                [tuple(map(int, e)) for e in bonds.t().tolist()],
                [int(x) for x in element],
            )
        return cache[key]

    def _with_hydrogens(self, coords, feats):
        if not getattr(self, "use_hydrogens", False):
            return coords
        if _HYDROGEN_OVERRIDE is not None:
            return _HYDROGEN_OVERRIDE
        ref = feats.get(self._H_REF_KEY)
        geom = feats.get(self._H_GEOM_KEY)
        if ref is None or geom is None or ref[0].shape[-1] == 0:
            return coords
        plan = self._hydrogen_plan_for(feats)
        r = geom[0][:, 0]  # bond lengths; the stored angles are unused
        phis = None
        if plan.get("rotor"):
            group = self.__dict__.get("_hydrogen_group")
            if group is None:
                group = self._hydrogen_group = _HydrogenGroup([self])
            phis = group.rotamers(coords, feats, plan, r)
        return _place_hydrogens(coords, plan, r, phis)

    def _select_terms(self, feats, index, *values):
        """Keep every term when hydrogens are rebuilt; otherwise only the terms
        whose atoms are all real atoms.

        Both extractions keep hydrogen-containing torsions, so without this the
        heavy-only mode indexes past the end of the coordinate tensor.
        """
        if getattr(self, "use_hydrogens", False):
            return index, values
        n_atoms = int(feats["atom_pad_mask"].shape[1])
        keep = (index < n_atoms).all(dim=0)
        return index[:, keep], tuple(v[keep] for v in values)


# --------------------------------------------------------------------------
# GAFF2 / OpenFF terms
# --------------------------------------------------------------------------


class GAFFBondPotential(AutogradGradientMixin, Potential):
    """GAFF2 bond lengths, Huber in |r - r0|.

    Quadratic within ``cutoff`` (0.2 A) and linear beyond, so the force is
    bounded by ``2 k cutoff``. With uniform_k=False -- what the gaff2 and
    openff-2.2.1 presets build -- k is the force field's own constant in
    kcal/mol/A^2 (AMBER's fc, which already folds in the 1/2; e.g. 232.5 for a
    GAFF2 sp3 C-C), not normalized.

    The bound is the point: guidance is plain descent, which is unstable for
    stiff (aromatic, polar) bonds at w * 8k > 2, and the cutoff bounds that
    oscillation. 0.5 let bonds swing to the break limit (measured: broken
    samples 77-98 at 0.2 against 2554-4586 at 0.5, 15 ligands).
    """

    def __init__(self, parameters=None, cutoff: float = 0.2,
                 uniform_k: bool = True) -> None:
        super().__init__(parameters=parameters)
        self.uniform_k = uniform_k
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive.")
        self.cutoff = cutoff

    def compute(self, coords, feats, parameters=None):
        index = feats["gaff_bond_index"][0]
        if index.shape[-1] == 0:
            return _connected_zero(coords)
        r0 = feats["gaff_bond_r0"][0]
        if self.uniform_k:
            k = torch.ones_like(r0)
        else:
            # GAFF2's own constant, kcal/mol/A^2. No conversion: AMBER folds
            # the 1/2 into fc and our Huber is k d^2, so fc is k.
            fc = feats.get("gaff_bond_fc")
            k = torch.ones_like(r0) if fc is None else fc[0]
        delta = (
            coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        )
        value = torch.sqrt((delta * delta).sum(-1) + 1e-12)
        dev = (value - r0).abs()
        d = self.cutoff
        energy = k * torch.where(dev <= d, dev * dev, d * (2.0 * dev - d))
        return energy.sum(dim=-1)


class GAFFAnglePotential(AutogradGradientMixin, Potential):
    """GAFF2 bond angles, ``k (theta - theta0)^2``.

    With uniform_k=False -- what the gaff2 and openff-2.2.1 presets build -- k
    is the force field's own constant in kcal/mol/rad^2, NOT normalized. Those
    constants carry the BALANCE of stiffness around a ring, which a uniform k
    erases -- and a five-ring's pucker is set by that balance.

    theta is computed with atan2 rather than acos: acos' is unbounded at a
    straight angle, where a nitrile or an alkyne sits exactly. ``cutoff``
    (0.35 rad in the presets) additionally makes the term Huber, bounding the
    force at 2 k cutoff.
    """

    def __init__(self, parameters=None, uniform_k: bool = True,
                 cutoff: Optional[float] = None) -> None:
        super().__init__(parameters=parameters)
        self.uniform_k = uniform_k
        if cutoff is not None and (not math.isfinite(cutoff) or cutoff <= 0):
            raise ValueError("cutoff must be finite and positive.")
        self.cutoff = cutoff

    def compute(self, coords, feats, parameters=None):
        index = feats["gaff_angle_index"][0]
        if index.shape[-1] == 0:
            return _connected_zero(coords)
        theta0 = feats["gaff_angle_theta0"][0]
        if self.uniform_k:
            k = torch.ones_like(theta0)
        else:
            fc = feats.get("gaff_angle_fc")
            k = torch.ones_like(theta0) if fc is None else fc[0]
        value = _angle_value(coords, index)
        delta = value - theta0
        if self.cutoff is None:
            energy = k * delta * delta
        else:
            # Huber: quadratic inside the cutoff, linear outside, so the force
            # saturates at 2 k cutoff instead of growing with the deviation.
            # This is the only term of ours whose force would otherwise be
            # unbounded -- MMFF's angle is a cosine well and both bonds are
            # already Huber.
            d = self.cutoff
            ad = delta.abs()
            energy = k * torch.where(ad <= d, ad * ad, d * (2.0 * ad - d))
        return energy.sum(dim=-1)


class GAFFTorsionPotential(AutogradGradientMixin, HydrogenMixin, Potential):
    """GAFF2 torsions: ``sum_n fc_n cos(n phi - phi0)`` for n = 1..6.

    Seven Fourier coefficients per torsion against MMFF's three, and the phase
    enters as ``cos(n phi - phi0)`` rather than ``cos(n (phi - phi0))``. The
    n = 0 coefficient is a constant offset and is dropped: it shifts the energy
    without changing any gradient.

    Unlike bond and angle this keeps GAFF2's own force constants: their spread
    across torsion types is exactly the chemistry the term is for.
    """

    _H_REF_KEY = "gaff_hydrogen_ref"
    _H_GEOM_KEY = "gaff_hydrogen_geom"
    _H_BOND_KEY = "gaff_bond_index"

    def __init__(self, parameters=None, use_hydrogens: bool = False) -> None:
        super().__init__(parameters=parameters)
        self.use_hydrogens = use_hydrogens

    def compute(self, coords, feats, parameters=None):
        coords = self._with_hydrogens(coords, feats)
        index, (phi0, fc) = self._select_terms(
            feats,
            feats["gaff_torsion_index"][0],
            feats["gaff_torsion_phi0"][0],
            feats["gaff_torsion_fc"][0],
        )
        if index.shape[-1] == 0:
            return _connected_zero(coords)
        value = _dihedral_value(coords, index)
        energy = value.new_zeros(value.shape)
        for n in range(1, 7):
            energy = energy + fc[:, n] * torch.cos(n * value - phi0)
        return energy.sum(dim=-1)


class GAFF14Potential(AutogradGradientMixin, HydrogenMixin, Potential):
    """The 1-4 nonbonded pairs, with a bounded repulsion.

    A/r^12 - B/r^6 + 332.0637 q_i q_j / (D r) over the pairs the force field
    marks as 1-4, with its own scaling already applied. These are part of what
    GAFF's torsion parameters were fitted against, so the torsion term is
    incomplete without them.

    Electrostatics are real Coulomb outside s = max(frac * size, 0.5 A) and a
    parabola matching value and slope inside (_coulomb_core); the van der Waals
    is the force field's own form outside its own switch and a parabola inside.
    """

    _H_REF_KEY = "gaff_hydrogen_ref"
    _H_GEOM_KEY = "gaff_hydrogen_geom"
    _H_BOND_KEY = "gaff_bond_index"

    _NORM_EPS = 1e-12  # keeps d||r||/dr finite at coincidence

    def __init__(self, parameters=None, use_hydrogens: bool = False,
                 dielectric: float = 1.0, lj_core_switch: float = 0.5,
                 coulomb_core_switch: float = 0.5) -> None:
        super().__init__(parameters=parameters)
        self.use_hydrogens = use_hydrogens
        for name, v in (("lj_core_switch", lj_core_switch),
                        ("coulomb_core_switch", coulomb_core_switch),
                        ("dielectric", dielectric)):
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        self.lj_core_switch = float(lj_core_switch)
        self.coulomb_core_switch = float(coulomb_core_switch)
        self.dielectric = float(dielectric)

    def compute(self, coords, feats, parameters=None):
        coords = self._with_hydrogens(coords, feats)
        index, (aij, bij, qij) = self._select_terms(
            feats,
            feats["gaff_pair_index"][0],
            feats["gaff_pair_aij"][0],
            feats["gaff_pair_bij"][0],
            feats["gaff_pair_qij"][0],
        )
        if index.shape[-1] == 0:
            return _connected_zero(coords)

        delta = (
            coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        )
        # Softened norm: ||delta|| has gradient delta/||delta||, which is 0/0 at
        # coincidence and gives NaN. Hydrogens are placed deterministically, so
        # two of them sharing internal coordinates would coincide exactly rather
        # than merely improbably.
        r = torch.sqrt((delta * delta).sum(-1) + self._NORM_EPS)

        # sigma from the pair's own A and B: A/B = sigma^6 where both are set.
        ratio = aij / bij.clamp_min(1e-12)
        sigma = torch.where(
            bij > 0, ratio.clamp_min(1e-12) ** (1.0 / 6.0),
            torch.full_like(ratio, 3.0)
        )
        # Pairs without Lennard-Jones (B = 0) get the floor as their switch.
        size = torch.where(bij > 0, sigma, torch.zeros_like(sigma))
        repulsive = _lj126_core(
            r, aij, bij, _core_switch(size, self.lj_core_switch)
        )
        coulomb = _coulomb_core(
            r,
            _GAFF_COULOMB * qij / self.dielectric,
            _core_switch(size, self.coulomb_core_switch),
        )
        return (repulsive + coulomb).sum(dim=-1)


class NonbondedPotential(AutogradGradientMixin, HydrogenMixin, Potential):
    """Full intramolecular nonbonded: the force field's own LJ and Coulomb.

        E(r) = lj(r) + coul(r)

    Both terms are bounded at coincidence by their core switch, so a clashing
    early structure gives a finite energy and a finite force.

    Scope: 1-2, 1-3 and 1-4 pairs are all excluded, the 1-4s because they are a
    separate term with the force field's own scaling. For a small ligand that
    leaves very few pairs, so this matters for macrocycles and peptides rather
    than for small rings.
    """

    _H_REF_KEY = "gaff_hydrogen_ref"
    _H_GEOM_KEY = "gaff_hydrogen_geom"
    _H_BOND_KEY = "gaff_bond_index"

    _EPS = 1e-12

    def __init__(self, parameters=None, use_hydrogens: bool = False,
                 dielectric: float = 1.0, lj_core_switch: float = 0.5,
                 coulomb_core_switch: float = 0.5) -> None:
        super().__init__(parameters=parameters)
        self.use_hydrogens = use_hydrogens
        for name, v in (("lj_core_switch", lj_core_switch),
                        ("coulomb_core_switch", coulomb_core_switch),
                        ("dielectric", dielectric)):
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        self.lj_core_switch = float(lj_core_switch)
        self.coulomb_core_switch = float(coulomb_core_switch)
        self.dielectric = float(dielectric)

    def compute(self, coords, feats, parameters=None):
        coords = self._with_hydrogens(coords, feats)
        index, (sigma, eps, qq) = self._select_terms(
            feats,
            feats["gaff_nb_index"][0],
            feats["gaff_nb_sigma"][0],
            feats["gaff_nb_eps"][0],
            feats["gaff_nb_qq"][0],
        )
        if index.shape[-1] == 0:
            return _connected_zero(coords)

        delta = (
            coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        )
        r = torch.sqrt((delta * delta).sum(-1) + self._EPS)
        sig = sigma.clamp_min(1e-6)
        # Pairs without Lennard-Jones (eps = 0) get the floor as their switch.
        size = torch.where(eps > 0, sig, torch.zeros_like(sig))
        s6 = sig**6
        vdw = _lj126_core(
            r,
            4.0 * eps * s6 * s6,
            4.0 * eps * s6,
            _core_switch(size, self.lj_core_switch),
        )
        coul = _coulomb_core(
            r,
            _GAFF_COULOMB * qq / self.dielectric,
            _core_switch(size, self.coulomb_core_switch),
        )
        return (vdw + coul).sum(dim=-1)


# --------------------------------------------------------------------------
# MMFF94 terms
# --------------------------------------------------------------------------


class MMFFBondPotential(AutogradGradientMixin, Potential):
    """MMFF-typed restraint on every ligand bond length.

    Huber in the length deviation: ``k dr^2`` within ``cutoff`` of r0 and linear
    beyond, so the curvature at the minimum is MMFF's own force constant while
    the force stays bounded by ``2 k cutoff``.

    The bound is the point. Guidance takes num_gd_steps descent steps per
    denoising step, and descent on a quadratic of curvature k diverges for a
    step above 2/k; with MMFF's force constant and no bound this blew every
    ligand out to 14-24 A bonds.

    MMFF's quartic correction is dropped for simplicity, not for correctness:
    with cs = -2 the shape factor has no stationary point but dr = 0, so it is
    monotonic either side of the minimum. It simply has no bounded-force region.
    """

    def __init__(self, parameters=None, cutoff: float = 0.2,
                 uniform_k: bool = False) -> None:
        super().__init__(parameters=parameters)
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be finite and positive.")
        self.cutoff = cutoff
        self.uniform_k = uniform_k

    def compute(self, coords, feats, parameters=None):
        index = feats["mmff_bond_index"][0]
        if index.shape[-1] == 0:
            return _connected_zero(coords)
        r0 = feats["mmff_bond_r0"][0]
        kb = feats.get("mmff_bond_kb")
        k = (
            torch.ones_like(r0)
            if (self.uniform_k or kb is None)
            else kb[0]
        )
        delta = (
            coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        )
        value = torch.sqrt((delta * delta).sum(-1) + 1e-12)
        dev = (value - r0).abs()
        d = self.cutoff
        energy = k * torch.where(dev <= d, dev * dev, d * (2.0 * dev - d))
        return energy.sum(dim=-1)


class MMFFAngleWellPotential(AutogradGradientMixin, Potential):
    """MMFF-typed restraint on every ligand bond angle.

    ``E = k (1 - cos(theta - theta0))``, with theta0 from MMFF94's own
    angle-bend parameters. The force is zero at theta0 and bounded by k
    everywhere, and at theta0 = 180 degrees the form reduces to MMFF's linear
    bend, so linear geometries need no special case.

    This is a well, not a flat bottom. A flat-bottom angle term cannot forbid a
    planar six-ring at any tolerance: for a planar hexagon the 1-3 distance is
    exactly sqrt(3) times the bond, so bond and 1-3 windows that admit the real
    ring also admit the flat one.
    """

    def __init__(self, parameters=None, uniform_k: bool = False) -> None:
        super().__init__(parameters=parameters)
        self.uniform_k = uniform_k

    def compute(self, coords, feats, parameters=None):
        index = feats["mmff_angle_index"][0]
        if index.shape[-1] == 0:
            return _connected_zero(coords)
        theta0 = feats["mmff_angle_theta0"][0]
        ka = feats.get("mmff_angle_ka")
        k = (
            torch.ones_like(theta0)
            if (self.uniform_k or ka is None)
            else ka[0]
        )
        value = _angle_value(coords, index)
        return (k * (1.0 - torch.cos(value - theta0))).sum(dim=-1)


class MMFFTorsionPotential(AutogradGradientMixin, HydrogenMixin, Potential):
    """Three-term MMFF94 torsion on ligand dihedrals.

    ``E = 1/2 [V1 (1 + cos phi) + V2 (1 - cos 2 phi) + V3 (1 + cos 3 phi)]``
    """

    _H_REF_KEY = "mmff_hydrogen_ref"
    _H_GEOM_KEY = "mmff_hydrogen_geom"
    _H_BOND_KEY = "mmff_bond_index"

    def __init__(self, parameters=None, use_hydrogens: bool = False) -> None:
        super().__init__(parameters=parameters)
        self.use_hydrogens = use_hydrogens

    def compute(self, coords, feats, parameters=None):
        coords = self._with_hydrogens(coords, feats)
        index, (v1, v2, v3) = self._select_terms(
            feats,
            feats["mmff_torsion_index"][0],
            feats["mmff_torsion_v1"][0],
            feats["mmff_torsion_v2"][0],
            feats["mmff_torsion_v3"][0],
        )
        if index.shape[-1] == 0:
            return _connected_zero(coords)
        value = _dihedral_value(coords, index)
        energy = 0.5 * (
            v1 * (1.0 + torch.cos(value))
            + v2 * (1.0 - torch.cos(2.0 * value))
            + v3 * (1.0 + torch.cos(3.0 * value))
        )
        return energy.sum(dim=-1)


class MMFF14Potential(AutogradGradientMixin, HydrogenMixin, Potential):
    """MMFF 1-4 pairs: buffered 14-7 vdW plus real Coulomb, both core-switched.

    MMFF's torsion parameters were fitted with these present, the same way
    GAFF's were, so the torsion term alone is incomplete in both.
    """

    _H_REF_KEY = "mmff_hydrogen_ref"
    _H_GEOM_KEY = "mmff_hydrogen_geom"
    _H_BOND_KEY = "mmff_bond_index"

    _NORM_EPS = 1e-12

    def __init__(self, parameters=None, use_hydrogens: bool = False,
                 dielectric: float = 1.0, lj_core_switch: float = 0.5,
                 coulomb_core_switch: float = 0.5) -> None:
        super().__init__(parameters=parameters)
        self.use_hydrogens = use_hydrogens
        for name, v in (("lj_core_switch", lj_core_switch),
                        ("coulomb_core_switch", coulomb_core_switch),
                        ("dielectric", dielectric)):
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        self.lj_core_switch = float(lj_core_switch)
        self.coulomb_core_switch = float(coulomb_core_switch)
        self.dielectric = float(dielectric)

    def _pair_keys(self):
        return ("mmff_pair_index", "mmff_pair_rstar", "mmff_pair_eps",
                "mmff_pair_qq")

    def compute(self, coords, feats, parameters=None):
        coords = self._with_hydrogens(coords, feats)
        ki, kr, ke, kq = self._pair_keys()
        index, (rstar, eps, qq) = self._select_terms(
            feats, feats[ki][0], feats[kr][0], feats[ke][0], feats[kq][0]
        )
        if index.shape[-1] == 0:
            return _connected_zero(coords)
        delta = (
            coords.index_select(-2, index[0]) - coords.index_select(-2, index[1])
        )
        r = torch.sqrt((delta * delta).sum(-1) + self._NORM_EPS)
        rs = rstar.clamp_min(1e-6)
        # MMFF's buffered 14-7 outside the core switch, a parabola matching
        # value and slope inside it.
        vdw = _mmff147_core(r, rs, eps, _core_switch(rs, self.lj_core_switch))
        coulomb = _coulomb_core(
            r,
            _MMFF_COULOMB * qq / self.dielectric,
            _core_switch(rs, self.coulomb_core_switch),
        )
        return (vdw + coulomb).sum(dim=-1)


class MMFFNonbondedPotential(MMFF14Potential):
    """Full intramolecular nonbonded for MMFF: core-switched buffered 14-7 plus
    real Coulomb, over the pairs that are not 1-2, 1-3 or 1-4.

    The vdW is deliberately the same form as ``MMFF14Potential``: modelling
    MMFF's vdW two different ways inside one potential set would be wrong
    whatever it bought elsewhere.

    MMFF's bare 332 q q / (r + 0.05) reaches ~6600 q q kcal/mol at coincidence,
    which is a hazard across the full nonbonded set because early diffusion
    structures do have overlapping atoms. The core switch bounds it instead.
    """

    def _pair_keys(self):
        return ("mmff_nb_index", "mmff_nb_rstar", "mmff_nb_eps", "mmff_nb_qq")


# --------------------------------------------------------------------------
# Benchmarked defaults
# --------------------------------------------------------------------------

#: The per-term weights the ligand presets steer with, measured on a 15-ligand
#: benchmark against 30 ns of GAFF2 MD (2026-11-23). The earlier values were
#: tuned against force magnitudes and pass rates; these are tuned against MD
#: geometry.
#:
#: Guidance is fixed-step gradient descent on the weighted energy, so a weight
#: IS that term's step length, not a strength. Descent on a bond of force
#: constant k diverges once w * 8k > 2, i.e. w* = 0.00038 for a GAFF2 C=O and
#: 0.00027 for an MMFF one, so the previous bond/angle 0.001 was past the bound
#: for every bond but sp3 C-C and was held together only by the Huber cutoff.
#: The damage sorted by force constant, not by molecule: at bond/angle 0.00125,
#: benzene reached 80.6% bad bonds while the three pure-sp3 rings stayed at
#: 0.00%.
#:
#: Bond, angle and torsion are a quarter of the earlier set. A uniform scale
#: preserves the ratios -- the fixed point of sum_i w_i grad U_i = 0, and so the
#: target geometry -- and shrinks only the step; scaling bond, or bond+angle,
#: ALONE measured worse than the old default, because the ratio is what sets the
#: geometry. pair14 is raised to the torsion weight because the benchmark held
#: the three soft terms together at 0.00125.
#:
#: Measured over 14 molecules x 3 seeds x 840 frames: bad bonds 6.3% -> 0.00%,
#: bond deviation 4.5% -> 2.0% against MD's own 1.6% floor, ring pucker Q 0.465
#: against MD's 0.459.
LIGAND_STEER_BOND_WEIGHT = 0.00025
LIGAND_STEER_ANGLE_WEIGHT = 0.00025
LIGAND_STEER_PAIR14_WEIGHT = 0.00125
LIGAND_STEER_TORSION_WEIGHT = 0.00125

#: Held at the torsion weight for every preset, OpenFF included through the
#: fallback. That is a uniform quarter of GAFF2's old 0.005 but only a sixth of
#: MMFF's old 0.0075, so MMFF's nonbonded term is driven 1.5x less than a
#: uniform rescale of its own preset would give. This value is the tested one
#: (0.04-0.05% bad bonds under MMFF); the uniform-quarter alternative for MMFF,
#: 0.001875, has not been run.
LIGAND_STEER_NONBONDED_WEIGHT = {"gaff2": 0.00125, "mmff": 0.00125}
LIGAND_STEER_NONBONDED_WEIGHT_DEFAULT = 0.00125

#: Every force-field term acts only for t <= this (t = 1 noise, 0 clean), in
#: guidance and in resampling. Early predictions overlap -- pairs inside
#: max(sigma/2, 0.5 A) up to t ~ 0.6, none from t = 0.5 -- and their bond and
#: angle energies collapsed FK resampling (bond 10.7, angle 6.7 log units of
#: spread across 1000 particles at t ~ 1, phenolate), driving the effective
#: sample size to ~1 at nearly every step.
#: When bond_angle_late_weight is used without its own t_on: the bond and
#: angle guidance weights rise to that value for t <= this, the last quarter
#: of the guided window.
LIGAND_STEER_BOND_ANGLE_LATE_T_ON = 0.1

LIGAND_STEER_T_ON = 0.4

#: Langevin guidance: x0 <- x0 - sum w grad E + sqrt(2 w kT) xi, with kT at
#: room temperature in kcal/mol. The kick is added once per descent step, INSIDE
#: the inner loop, so the noise accumulates as sqrt(N) while the drift
#: accumulates linearly: at LIGAND_STEER_GD_STEPS this weight is about 0.17 A of
#: thermal displacement, where the old 0.005 would be 0.35 A. (Those were 0.39
#: and 0.77 A when this constant was 100; both scale as sqrt(steps).)
#:
#: This is a trade, not an ordering. Measured at bond/angle 0.00025, gd 80,
#: as bioemu2 / boltz1:
#:
#:   langevin   bad bonds       pucker Q (MD 0.459)   boltz1 mean |dtors|
#:   0.005      0.27% / 0.60%   0.311 / 0.331         10.0 deg
#:   0.00125    0.00% / 0.00%   0.460 / 0.465         20.4 deg
#:
#: The cooler value is the default because it is the one that leaves bonds and
#: ring geometry intact; raise it to 0.005 when conformational spread matters
#: more than bond lengths. One counterintuitive detail: where the descent is
#: ALREADY unstable, more noise does less bond damage (19.8% vs 45.7% at bond
#: 0.00125), because the kick decoheres the overshoot instead of compounding it.
LIGAND_STEER_LANGEVIN_KT = 0.593
LIGAND_STEER_LANGEVIN_WEIGHT = 0.00125

#: Particles carried per requested sample, and also the size of each independent
#: resampling group: the presets set the group size TO this value, so each output
#: sample's particles compete only with each other instead of in one global pool
#: over the whole batch. A single global pool measured better at large pool
#: sizes, but the benchmarked setting is k = 3 grouped, which keeps the cost per
#: sample fixed and still carries the FK selection.
LIGAND_STEER_PARTICLES_PER_SAMPLE = 3

#: Inner gradient-descent steps per diffusion step. The gradient is re-evaluated
#: at the displaced geometry on every iteration, so this is genuine iterative
#: descent, and the soft terms need the steps that the x0.25 weights cost them:
#: it is w * steps that reaches a fixed point. 80 and 20 were benchmarked -- in
#: boltz1 they tie, in bioemu2 20 loses the open conformers (bipyridine N-C-C-N
#: 122 deg -> 45, against MD's 166). 100 extrapolated the better of the two and
#: was never itself benchmarked.
#:
#: Set to 20 by request, 2026-11-25, with that bioemu2 result known and
#: accepted. Two things follow, neither of them only a cost saving:
#:   - bioemu2 is expected to close the open conformers again, as measured.
#:   - the Langevin kick is applied once per descent step, so its accumulated
#:     thermal displacement falls with sqrt(N): about 0.17 A here against the
#:     0.39 A of 100 steps. Fewer steps is also colder.
#: A later sweep over gd 20-100 found the aggregate error flat (mean 48.9 ->
#: 41.5 deg) with 0% broken bonds throughout, which is why 20 is defensible
#: despite the above; the aggregate is dominated by wrong-basin cells that gd
#: does not move either way.
#: Raise it back to 100 with -S num_gd_steps=100, or revert this constant.
LIGAND_STEER_GD_STEPS = 20

#: vdW core switch fraction, by force field. See _core_switch.
LIGAND_STEER_LJ_CORE_SWITCH = {"gaff2": 0.7, "mmff": 0.5}
LIGAND_STEER_COULOMB_CORE_SWITCH = 0.5

#: Bond Huber cutoff in A, and the angle Huber cutoff in rad for GAFF/OpenFF.
LIGAND_STEER_BOND_CUTOFF = 0.2
LIGAND_STEER_ANGLE_CUTOFF = 0.35

#: The physicality terms that override a ligand force field: they restrain the
#: same geometry at 10-50x the force field's own weights. Measured on the
#: 15-ligand set, dropping them is neutral for MMFF (0.0% wrong either way) and
#: helps GAFF2 (16.7 -> 10.0%), while OpenFF is reproducibly worse without them
#: (12.5 -> 25.0%, SD 0.0 over three seeds).
#: PoseBusters is KEPT: a flat bottom on RDKit's distance bounds, so within
#: +-20% of a bond's bound it exerts no force and the force field alone decides
#: the geometry, while gross failures stay bounded. BondLength and BondAngle are
#: bioemu2 names with no counterpart here, listed so both forks drop one set.
LIGAND_PRESET_DROP = (
    "BondLengthPotential",
    "BondAnglePotential",
)

#: Which preset names select which force field.
GAFF_PRESETS = ("gaff2", "openff-2.2.1", "openff-2.1.0", "openff-2.0.0")
MMFF_PRESETS = ("mmff",)
LIGAND_PRESETS = (*GAFF_PRESETS, *MMFF_PRESETS)


def _late_schedule(weight, *t_ons):
    """weight while t <= the smallest given t_on (late in denoising, t = 0
    clean), 0 before; the weight unchanged when every t_on is None."""
    vals = [float(t) for t in t_ons if t is not None]
    if not vals:
        return weight
    return PiecewiseStepFunction(thresholds=[min(vals)], values=[weight, 0.0])


_NONBONDED_TYPES = (
    GAFF14Potential,
    NonbondedPotential,
    MMFF14Potential,
    MMFFNonbondedPotential,
)


def _apply_t_on(terms, t_on, nonbonded_t_on=None, guidance_ramp: bool = False):
    """Every force-field term acts (guidance and resampling) only for
    t <= t_on; the 1-4 and nonbonded terms also only for t <= nonbonded_t_on
    when given.

    With guidance_ramp the guidance weight instead rises linearly from 0 at
    t = 1 to its full value at t = 0; resampling keeps the t_on switch.
    """
    for p in terms:
        nb = isinstance(p, _NONBONDED_TYPES)
        limits = (t_on, nonbonded_t_on) if nb else (t_on,)
        params = dict(p.parameters or {})
        for key in ("guidance_weight", "resampling_weight"):
            if key not in params:
                continue
            if key == "guidance_weight" and guidance_ramp:
                params[key] = LinearInterpolation(
                    start_t=1.0, end_t=0.0, start_val=0.0, end_val=params[key]
                )
            else:
                params[key] = _late_schedule(params[key], *limits)
        p.parameters = params
        known = [float(x) for x in limits if x is not None]
        p.t_on = min(known) if known else None
    return terms


_BOND_TYPES = (GAFFBondPotential, MMFFBondPotential)
_ANGLE_TYPES = (GAFFAnglePotential, MMFFAngleWellPotential)


def _apply_late_bond_angle(terms, t_on, late_t_on, late_bond, late_angle):
    """Raise the bond and angle GUIDANCE weights for t <= late_t_on.

    Bond and angle are stiff, local and nearly convex, so clamping them hard
    only at the end -- once the torsions that decide chemical state have
    settled, and with little noise left to wash the correction out -- pulls
    back the bond stretch that a large 1-4 weight induces without disturbing
    the conformation that weight was raised to get.

    Resampling weights are deliberately left alone: changing the Feynman-Kac
    reward partway through a run would reweight particles for a different
    objective than the one they were grown under.
    """
    for p in terms:
        if isinstance(p, _BOND_TYPES):
            late = late_bond
        elif isinstance(p, _ANGLE_TYPES):
            late = late_angle
        else:
            continue
        if late is None:
            continue
        params = dict(p.parameters or {})
        early = params.get("guidance_weight")
        if early is None:
            continue
        if isinstance(early, PiecewiseStepFunction):
            # _apply_t_on already made a step at t_on; rebuild from its value.
            early = early.values[0]
        elif not isinstance(early, (int, float)):
            raise ValueError(
                "a late bond/angle weight needs a plain weight to start from, "
                f"got {type(early).__name__}"
            )
        thresholds = [late_t_on] if t_on is None else [late_t_on, t_on]
        values = [late, early] if t_on is None else [late, early, 0.0]
        params["guidance_weight"] = PiecewiseStepFunction(
            thresholds=thresholds, values=values
        )
        p.parameters = params
    return terms


#: Which physicality terms a ligand preset leaves out. BondLength and BondAngle
#: are bioemu2 names with no counterpart here, so "bond_angle" and "none" differ
#: only in bioemu2; "all" additionally drops PoseBusters, which does exist here,
#: and is what makes selection pressure purely force-field in both forks.
LIGAND_DROP_SETS = {
    "none": (),
    "bond_angle": LIGAND_PRESET_DROP,
    "all": ("PoseBustersPotential", *LIGAND_PRESET_DROP),
}


def _drop_overriding_physicality(potentials, which="bond_angle"):
    """The physicality set without the terms named by LIGAND_DROP_SETS."""
    if which not in LIGAND_DROP_SETS:
        msg = (f"drop_overriding takes one of {', '.join(LIGAND_DROP_SETS)}; "
               f"got {which!r}")
        raise ValueError(msg)
    drop = LIGAND_DROP_SETS[which]
    return [p for p in potentials if type(p).__name__ not in drop]


def get_ligand_potentials(
    steering_args,
    preset: str,
    *,
    boltz2: bool = True,
    bond: bool = True,
    angle: bool = True,
    torsion: bool = True,
    pairs14: bool = True,
    nonbonded: bool = True,
    hydrogen: bool = True,
    physicality: bool = True,
    drop_overriding: str = "bond_angle",
    bond_weight: Optional[float] = None,
    angle_weight: Optional[float] = None,
    torsion_weight: Optional[float] = None,
    pair14_weight: Optional[float] = None,
    nonbonded_weight: Optional[float] = None,
    bond_cutoff: float = LIGAND_STEER_BOND_CUTOFF,
    angle_cutoff: Optional[float] = None,
    bond_uniform_k: Optional[bool] = None,
    angle_uniform_k: Optional[bool] = None,
    dielectric: float = 1.0,
    pairs14_dielectric: Optional[float] = None,
    nonbonded_dielectric: Optional[float] = None,
    lj_core_switch: Optional[float] = None,
    coulomb_core_switch: float = LIGAND_STEER_COULOMB_CORE_SWITCH,
    t_on: Optional[float] = LIGAND_STEER_T_ON,
    nonbonded_t_on: Optional[float] = None,
    guidance_ramp: bool = False,
    bond_angle_late_weight: Optional[float] = None,
    bond_late_weight: Optional[float] = None,
    angle_late_weight: Optional[float] = None,
    bond_angle_late_t_on: Optional[float] = None,
):
    """Boltz's physicality set plus the ligand force-field terms of ``preset``.

    Defaults are the benchmarked ones (LIGAND_STEER_*). ``preset`` is "mmff" or
    a GAFF/OpenFF name; the GAFF classes carry OpenFF too, since OpenFF shares
    GAFF2's functional forms.
    """
    if preset not in LIGAND_PRESETS:
        raise ValueError(
            f"unknown ligand force field {preset!r}; "
            f"choose from {', '.join(LIGAND_PRESETS)}"
        )
    is_mmff = preset in MMFF_PRESETS
    family = "mmff" if is_mmff else "gaff2"

    # hydrogen=True because the models predict heavy atoms only and the
    # heavy-only torsion path is INVERTED, not merely weaker -- see
    # HydrogenMixin.
    if bond_weight is None:
        bond_weight = LIGAND_STEER_BOND_WEIGHT
    if angle_weight is None:
        angle_weight = LIGAND_STEER_ANGLE_WEIGHT
    if torsion_weight is None:
        torsion_weight = LIGAND_STEER_TORSION_WEIGHT
    if pair14_weight is None:
        pair14_weight = LIGAND_STEER_PAIR14_WEIGHT
    if nonbonded_weight is None:
        # By preset, not by family: openff-2.2.1 takes the 0.0075 default, not
        # gaff2's 0.005.
        nonbonded_weight = LIGAND_STEER_NONBONDED_WEIGHT.get(
            preset, LIGAND_STEER_NONBONDED_WEIGHT_DEFAULT
        )
    if lj_core_switch is None:
        lj_core_switch = LIGAND_STEER_LJ_CORE_SWITCH[family]
    if bond_uniform_k is None:
        # Every preset uses the force field's own constants: GAFF2/OpenFF's fc
        # and MMFF's kb. A uniform k discards the balance of stiffness across
        # bond types, which is what sets a ring's shape.
        bond_uniform_k = False
    if angle_uniform_k is None:
        angle_uniform_k = False
    if angle_cutoff is None and not is_mmff:
        # The only term whose force would otherwise be unbounded; keeps
        # ordinary strain quadratic and bounds the force past 0.35 rad.
        angle_cutoff = LIGAND_STEER_ANGLE_CUTOFF

    pair_dielectric = dielectric if pairs14_dielectric is None else pairs14_dielectric
    nb_dielectric = dielectric if nonbonded_dielectric is None else nonbonded_dielectric

    def params(weight):
        return {
            "guidance_interval": 1,
            "guidance_weight": weight,
            "resampling_weight": 0.1,
        }

    pair_kw = dict(
        use_hydrogens=hydrogen,
        lj_core_switch=lj_core_switch,
        coulomb_core_switch=coulomb_core_switch,
    )

    extra = []
    if bond:
        cls = MMFFBondPotential if is_mmff else GAFFBondPotential
        extra.append(
            cls(cutoff=bond_cutoff, uniform_k=bond_uniform_k,
                parameters=params(bond_weight))
        )
    if angle:
        if is_mmff:
            extra.append(
                MMFFAngleWellPotential(
                    uniform_k=angle_uniform_k, parameters=params(angle_weight)
                )
            )
        else:
            extra.append(
                GAFFAnglePotential(
                    cutoff=angle_cutoff, uniform_k=angle_uniform_k,
                    parameters=params(angle_weight),
                )
            )
    if pairs14:
        cls = MMFF14Potential if is_mmff else GAFF14Potential
        extra.append(
            cls(dielectric=pair_dielectric, parameters=params(pair14_weight),
                **pair_kw)
        )
    if nonbonded:
        cls = MMFFNonbondedPotential if is_mmff else NonbondedPotential
        extra.append(
            cls(dielectric=nb_dielectric, parameters=params(nonbonded_weight),
                **pair_kw)
        )
    if torsion:
        cls = MMFFTorsionPotential if is_mmff else GAFFTorsionPotential
        extra.append(
            cls(use_hydrogens=hydrogen, parameters=params(torsion_weight))
        )

    _apply_t_on(extra, t_on, nonbonded_t_on, guidance_ramp)

    late_bond = (
        bond_late_weight if bond_late_weight is not None else bond_angle_late_weight
    )
    late_angle = (
        angle_late_weight if angle_late_weight is not None else bond_angle_late_weight
    )
    if late_bond is not None or late_angle is not None:
        late_t_on = bond_angle_late_t_on
        if late_t_on is None:
            late_t_on = LIGAND_STEER_BOND_ANGLE_LATE_T_ON
        if guidance_ramp:
            raise ValueError(
                "guidance_ramp replaces the whole guidance schedule; it cannot "
                "be combined with a late bond/angle weight"
            )
        if t_on is not None and late_t_on >= t_on:
            raise ValueError(
                f"bond_angle_late_t_on ({late_t_on}) must be below t_on ({t_on}); "
                "the late stage is the tail of the guided window, not all of it"
            )
        _apply_late_bond_angle(extra, t_on, late_t_on, late_bond, late_angle)

    _share_hydrogen_group(extra)

    base = list(get_potentials(steering_args, boltz2=boltz2)) if physicality else []
    if physicality:
        # A bool is how this read before the set became three-way: keep it
        # working so an older -P drop_overriding=0/1 means none/bond_angle.
        which = drop_overriding
        if isinstance(which, bool):
            which = "bond_angle" if which else "none"
        base = _drop_overriding_physicality(base, which)
    return [*base, *extra]


def steering_energies(potentials, coords, feats, t: float = 0.0):
    """Per-term UNWEIGHTED energy of every particle, as {term: tensor[B]}.

    Written into the output structure so a run can be scored without rerunning
    the model, the same way bioemu2 records its ``r_bioemu2_E_*`` properties.
    The values are the potentials' own energies in kcal/mol, NOT multiplied by
    their guidance weights: the weights are a steering choice, while these are
    the force field's verdict on the geometry.

    A term that contributes nothing to this structure reports 0.0 rather than
    being left out, so every sample carries the same set of names.
    """
    out = {}
    with torch.no_grad():
        for p in potentials:
            name = type(p).__name__.removesuffix("Potential")
            try:
                e = p.compute(coords, feats, p.compute_parameters(t) or {})
            except Exception as exc:  # noqa: BLE001 - must not fail a run
                # Loud, not silent: an energy that cannot be evaluated means
                # the term is misconfigured, and a quietly absent property
                # looks exactly like a term that is simply zero.
                logger.warning(
                    "could not evaluate %s for the output energies: %s",
                    type(p).__name__, exc,
                )
                continue
            if e is None:
                continue
            out[name] = e.detach().float().reshape(-1)
    if out:
        total = None
        for v in out.values():
            total = v.clone() if total is None else total + v
        out["total"] = total
    return out


def reconstructed_hydrogens(potentials, coords, feats):
    """The steering hydrogens for ``coords``, as (positions, parents).

    ``positions`` is [B, H, 3] and ``parents`` the parent atom index of each
    hydrogen, so a writer can add them to their parent's residue and chain and
    bond them to it. Returns (None, None) when nothing reconstructs hydrogens.

    These are the SAME positions the torsion and 1-4 terms were evaluated on --
    built from each parent's current heavy neighbours, with rotatable OH/SH at
    the rotamer the group chose -- so the written structure matches what the
    force field actually saw rather than a fresh guess.
    """
    for p in potentials:
        if not isinstance(p, HydrogenMixin) or not getattr(p, "use_hydrogens", False):
            continue
        ref = feats.get(p._H_REF_KEY)
        geom = feats.get(p._H_GEOM_KEY)
        if ref is None or geom is None or ref[0].shape[-1] == 0:
            continue
        try:
            with torch.no_grad():
                full = p._with_hydrogens(coords, feats)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not rebuild hydrogens for output: %s", exc)
            return None, None
        n_h = int(ref[0].shape[-1])
        if full.shape[-2] <= coords.shape[-2]:
            return None, None
        return full[..., -n_h:, :].detach(), ref[0].reshape(-1).tolist()
    return None, None


__all__ = [
    "GAFF14Potential",
    "GAFFAnglePotential",
    "GAFFBondPotential",
    "GAFFTorsionPotential",
    "GAFF_PRESETS",
    "LIGAND_PRESETS",
    "LIGAND_DROP_SETS",
    "LIGAND_PRESET_DROP",
    "LIGAND_STEER_ANGLE_WEIGHT",
    "LIGAND_STEER_BOND_WEIGHT",
    "LIGAND_STEER_GD_STEPS",
    "LIGAND_STEER_LANGEVIN_KT",
    "LIGAND_STEER_LANGEVIN_WEIGHT",
    "LIGAND_STEER_NONBONDED_WEIGHT",
    "LIGAND_STEER_NONBONDED_WEIGHT_DEFAULT",
    "LIGAND_STEER_PAIR14_WEIGHT",
    "LIGAND_STEER_PARTICLES_PER_SAMPLE",
    "LIGAND_STEER_T_ON",
    "LIGAND_STEER_TORSION_WEIGHT",
    "MMFF14Potential",
    "MMFFAngleWellPotential",
    "MMFFBondPotential",
    "MMFFNonbondedPotential",
    "MMFFTorsionPotential",
    "MMFF_PRESETS",
    "NonbondedPotential",
    "get_ligand_potentials",
    "reconstructed_hydrogens",
    "steering_energies",
]
