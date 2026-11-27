"""Rotational energy of the lowest rovibronic level (EA_0 between lowest levels).

EA_0 is measured between the lowest rovibronic levels: anion (v=0, lowest J) -> neutral
(v=0, lowest J).  The other layers give the rotationless energy of v=0 plus, for a
Lambda > 0 multiplet, the lowest spin-orbit component in first order (-|A| Lambda S).
The lowest level of a Lambda > 0 state carries rotational energy that a Sigma state does
not: in Hund's case (a) B [J(J+1) - Omega^2 + S(S+1) - Sigma^2] at J = |Omega|, i.e. +B for
a regular 2Pi_1/2 and +2B for an inverted 2Pi_3/2; for |A| ~ 2B (CH) the case (a) shift
is quenched by spin uncoupling.  This layer diagonalizes the standard effective
Hamiltonian A Lambda Sigma + B (J - L - S)^2 in the case (a) basis (Hill-Van Vleck for
2Pi) and returns the lowest level minus the shift the spin-orbit layer already applied.

OH X2Pi_3/2: lowest level J=3/2 at -37.7 cm-1 above the spin-orbit-free origin instead of
-A/2 = -69.6 cm-1 (+0.0040 eV on EA_0); CH X2Pi_1/2: +B = +14 cm-1 (+0.0018 eV).  Heavy
molecules (B < 0.5 cm-1): < 1e-4 eV.  Neglected: Lambda doubling, spin-rotation and
spin-spin splittings (< 1 cm-1 for the hydrides), B_0 - B_e (B from r_e, ~2 %).

Regular / inverted 2S+1Pi from the reference occupation (``pi_regularity``): the open pi
shell holds one electron (regular, A > 0) or three (inverted, A < 0).  When it cannot be
read (no irrep occupation, Delta states, several open pi shells) the central value is
the mean of both cases and the bound covers either.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

CM1_TO_EV = 1.0 / 8065.543937
# h / (8 pi^2 c) in cm-1 u Angstrom^2
ROTATIONAL_CONSTANT_FACTOR = 16.857629206
RELATIVE_BOUND = 0.2          # B_0 vs B_e, neglected fine structure
ABSOLUTE_BOUND_CM1 = 0.5      # Lambda doubling, spin-rotation, spin-spin
# C2v / D2h irrep pairs that carry the two components of a pi orbital (x, y).
PI_PAIRS = (("B1", "B2"), ("B2u", "B3u"), ("B2g", "B3g"), ("b1", "b2"), ("B2U", "B3U"), ("B2G", "B3G"))


def rotational_constant_cm1(mass_a_u: float, mass_b_u: float, r_angstrom: float) -> float:
    mu = mass_a_u * mass_b_u / (mass_a_u + mass_b_u)
    return ROTATIONAL_CONSTANT_FACTOR / (mu * r_angstrom ** 2)


def lowest_level_cm1(lam: int, two_s: int, a_cm1: float, b_cm1: float, extra_j: int = 4) -> float:
    """Lowest eigenvalue of A Lambda Sigma + B (J - L - S)^2 over J (case (a) basis,
    Omega = Lambda + Sigma, one sign of Lambda; Lambda doubling neglected).  Zero of energy:
    the rotationless, spin-orbit-free level.  Lambda = 0: 0 (N = 0)."""
    if lam == 0:
        return 0.0
    s = two_s / 2.0
    sigmas = [-s + k for k in range(two_s + 1)]
    omegas = [lam + sg for sg in sigmas]
    j_min = min(abs(o) for o in omegas)
    best = np.inf
    for step in range(extra_j + 1):
        j = j_min + step
        comps = [(sg, om) for sg, om in zip(sigmas, omegas) if abs(om) <= j + 1e-9]
        n = len(comps)
        h = np.zeros((n, n))
        for i, (sg, om) in enumerate(comps):
            h[i, i] = a_cm1 * lam * sg + b_cm1 * (j * (j + 1) - om ** 2 + s * (s + 1) - sg ** 2)
        for i in range(n - 1):
            (sg, om), (sg2, om2) = comps[i], comps[i + 1]
            if abs(sg2 - sg - 1.0) < 1e-9:
                h[i, i + 1] = h[i + 1, i] = -b_cm1 * np.sqrt(max(j * (j + 1) - om * (om + 1), 0.0)) \
                    * np.sqrt(max(s * (s + 1) - sg * (sg + 1), 0.0))
        best = min(best, float(np.linalg.eigvalsh(h)[0]))
    return best


def pi_regularity(irrep_nelec: Mapping[str, Sequence[int]] | None, lam: int) -> str | None:
    """REGULAR (pi^1-like open shell) or INVERTED (pi^3-like) for a Pi state from the
    (n_alpha, n_beta) per irrep; None when it cannot be read."""
    if lam != 1 or not irrep_nelec:
        return None
    occ = {str(k): (int(v[0]), int(v[1])) for k, v in irrep_nelec.items()}
    found = []
    for x, y in PI_PAIRS:
        if x not in occ or y not in occ:
            continue
        nx, ny = sum(occ[x]), sum(occ[y])
        if nx == ny:
            continue
        more, less = (x, y) if nx > ny else (y, x)
        spin_more = occ[more][0] - occ[more][1]
        spin_less = occ[less][0] - occ[less][1]
        if abs(nx - ny) != 1 or spin_more == spin_less:
            return None
        found.append("REGULAR" if spin_more > spin_less else "INVERTED")
    return found[0] if len(found) == 1 else None


def state_rotation_offset(*, lam: int, two_s: int, b_cm1: float, delta_so_ev: float | None,
                          regularity: str | None) -> dict:
    """Lowest rovibronic level minus the energy the other layers assign (rotationless v=0
    plus the first-order spin-orbit shift ``delta_so_ev`` of the lowest component)."""
    if lam == 0:
        return {"offset_cm1": 0.0, "bound_cm1": 0.0, "case": "SIGMA_N0"}
    s = two_s / 2.0
    applied = 0.0 if delta_so_ev is None else delta_so_ev / CM1_TO_EV
    a_abs = 0.0 if (delta_so_ev is None or s == 0) else abs(applied) / (lam * s)
    if a_abs == 0.0:
        off = lowest_level_cm1(lam, two_s, 0.0, b_cm1) - applied
        alt = off
        case = "NO_FIRST_ORDER_SO"
    else:
        reg = lowest_level_cm1(lam, two_s, +a_abs, b_cm1) - applied
        inv = lowest_level_cm1(lam, two_s, -a_abs, b_cm1) - applied
        if regularity == "REGULAR":
            off, alt, case = reg, reg, "REGULAR"
        elif regularity == "INVERTED":
            off, alt, case = inv, inv, "INVERTED"
        else:
            off, alt, case = 0.5 * (reg + inv), reg, "REGULARITY_UNKNOWN"
            alt = max(abs(reg - off), abs(inv - off))
    if case == "REGULARITY_UNKNOWN":
        bound = alt + RELATIVE_BOUND * abs(off) + ABSOLUTE_BOUND_CM1
    else:
        bound = RELATIVE_BOUND * abs(off) + ABSOLUTE_BOUND_CM1
    return {"offset_cm1": off, "bound_cm1": bound, "case": case, "a_abs_cm1": a_abs, "b_cm1": b_cm1}


def run_rotation_stage(*, electronic_ea: Mapping[str, Any] | None, requests: Sequence[Any],
                       nuclear: Mapping[str, Any] | None, spin_orbit: Mapping[str, Any] | None) -> dict:
    from .spin_orbit_stage import parse_term_symbol
    record: dict[str, Any] = {"stage": "LOWEST_ROVIBRONIC_LEVEL", "validation_metadata_used_in_computation": False}
    if not electronic_ea or electronic_ea.get("status") != "ELECTRONIC_EA_EVALUATED" or not nuclear:
        return {**record, "status": "NOT_ATTEMPTED", "reason": "electronic EA or nuclear motion missing"}
    masses = nuclear.get("masses") or {}
    if masses.get("mass_a_u") is None or masses.get("mass_b_u") is None:
        return {**record, "status": "NOT_ATTEMPTED", "reason": "no isotope masses in the nuclear record"}
    template = {}
    for req in requests:
        template.setdefault(req.job_id, req)
    so_states = (spin_orbit or {}).get("states") or {}
    so_ok = bool(spin_orbit) and spin_orbit.get("status") == "CLEARED"
    states = {}
    for role in ("neutral", "anion"):
        job_id = electronic_ea[f"selected_{role}_job"]
        req = template.get(job_id)
        cand = next(c for c in electronic_ea["candidates"][role] if c["job_id"] == job_id)
        term = parse_term_symbol(getattr(req, "symmetry_label", None) if req is not None else None)
        if term is None:
            return {**record, "status": "NOT_ATTEMPTED", "reason": f"{role}: no term symbol"}
        mult, lam = term
        re_ = (((nuclear.get("cross_model_identity_checks") or {}).get(role) or {}).get("re_primary_angstrom")
               or cand.get("r_central_angstrom"))
        b = rotational_constant_cm1(float(masses["mass_a_u"]), float(masses["mass_b_u"]), float(re_))
        so = so_states.get(role) or {}
        dso = so.get("delta_so_ev") if (so_ok and so.get("status") == "CLEARED") else None
        regularity = pi_regularity(getattr(req, "irrep_nelec", None), lam)
        states[role] = {"symmetry_label": req.symmetry_label, "lambda": lam, "two_s": mult - 1,
                        "r_angstrom": float(re_), "regularity": regularity,
                        **state_rotation_offset(lam=lam, two_s=mult - 1, b_cm1=b, delta_so_ev=dso,
                                                regularity=regularity)}
    delta = (states["neutral"]["offset_cm1"] - states["anion"]["offset_cm1"]) * CM1_TO_EV
    bound = (states["neutral"]["bound_cm1"] + states["anion"]["bound_cm1"]) * CM1_TO_EV
    return {**record, "status": "CLEARED", "states": states, "delta_rot_ev": delta, "bound_ev": bound}
