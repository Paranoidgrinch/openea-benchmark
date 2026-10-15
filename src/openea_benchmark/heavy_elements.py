"""Element rules beyond Kr: pseudopotential route (heavy-element step 1).

Every rule here is per element and molecule-independent.

Route (``element_route``):

* ALL_ELECTRON - Z <= 36, the existing all-electron chain (frozen core, aug-cc-pVXZ,
  CV in aug-cc-pwCVXZ, scalar relativity as sfX2C(DK basis) - NR).
* PSEUDOPOTENTIAL - Z > 36 when the correlation-consistent PP series
  aug-cc-pV{T,Q,5}Z-PP loads for the element (PySCF library, else Basis Set
  Exchange).  The ECP replaces the inner core and carries scalar relativity; the
  def2 bases of the scout/MR stages come with their own def2 ECPs.
* None - fail closed with a reason: open 4f/5f shells (La-Lu, Ac-Lr: strongly
  multireference, outside the CCSD(T)-based chain) and elements without a PP
  cardinal series (Rb, Sr, Cs, Ba have no aug-cc-pVXZ-PP in BSE).

Frozen core (``valence_core_orbitals``) counts all-electron spatial orbitals of the
conventional valence frozen core (Peterson's cc-pVXZ-PP practice): 4d metals
[Kr] (4s4p frozen, 4d5s correlated, like [Ar] for Sc-Zn), 5p elements [Kr]4d10,
5d metals [Xe]4f14, 6p elements [Xe]4f14 5d10.  With an ECP the frozen orbitals
are this count minus the ECP's core orbitals (``frozen_orbitals``), so the
correlated space is the same as in an all-electron calculation of the element.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

ROUTE_ALL_ELECTRON = "ALL_ELECTRON"
ROUTE_PSEUDOPOTENTIAL = "PSEUDOPOTENTIAL"
OPEN_F_BLOCK = frozenset(range(57, 72)) | frozenset(range(89, 104))
PP_ROUTE_SERIES = "aug-cc-pV{x}Z-PP"
PP_ROUTE_CARDINALS = (3, 4, 5)
CARDINAL_LETTER = {2: "D", 3: "T", 4: "Q", 5: "5", 6: "6"}

# Core shells in the order they are frozen; cumulative counts give the core sizes.
_CORE_SHELLS = ((1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2),
                (5, 0), (5, 1), (4, 3), (5, 2))
# (last Z of the range, frozen spatial orbitals); open f block handled separately.
_FROZEN_CORE_BY_RANGE = ((2, 0), (10, 1), (18, 5), (30, 9), (36, 14), (48, 18), (54, 23),
                         (56, 27), (80, 34), (86, 39))
# Valence shells of the full-valence CAS, by C2v irrep (a1, b1, b2, a2).
VALENCE_S, VALENCE_SP, VALENCE_SD = (1, 0, 0, 0), (2, 1, 1, 0), (3, 1, 1, 1)


def _z(element: str) -> int:
    from pyscf import gto
    return int(gto.charge(element))


def valence_core_orbitals(z: int) -> int | None:
    """All-electron spatial orbitals of the valence frozen core, None for open f shells."""
    if z in OPEN_F_BLOCK or z > 86:
        return None
    for last, n in _FROZEN_CORE_BY_RANGE:
        if z <= last:
            return n
    return None


def is_d_block(z: int) -> bool:
    return 21 <= z <= 30 or 39 <= z <= 48 or 72 <= z <= 80


def shell_irreps_c2v(l: int) -> tuple[int, int, int, int]:
    """C2v (a1, b1, b2, a2) counts of one shell of angular momentum l about the z axis."""
    counts = [1, 0, 0, 0]  # m = 0
    for m in range(1, l + 1):
        if m % 2:
            counts[1] += 1
            counts[2] += 1
        else:
            counts[0] += 1
            counts[3] += 1
    return tuple(counts)


def core_irreps_c2v(n_core: int) -> tuple[int, int, int, int] | None:
    """C2v irreps of the first ``n_core`` core orbitals (whole shells only)."""
    total, counts = 0, [0, 0, 0, 0]
    for _, l in _CORE_SHELLS:
        if total == n_core:
            return tuple(counts)
        counts = [a + b for a, b in zip(counts, shell_irreps_c2v(l))]
        total += 2 * l + 1
    return tuple(counts) if total == n_core else None


def valence_shells_c2v(z: int) -> tuple[int, int, int, int]:
    if z <= 2:
        return VALENCE_S
    return VALENCE_SD if is_d_block(z) else VALENCE_SP


# Shells an ECP replaces: ECP28 = [Ar]3d10, ECP60 = [Kr]4d10 4f14 (5s5p stay explicit).
_ECP_CORE_SHELLS = {14: _CORE_SHELLS[:6], 30: _CORE_SHELLS[:9] + ((4, 3),)}


def ecp_core_irreps_c2v(ecp_electrons: int) -> tuple[int, int, int, int] | None:
    if ecp_electrons == 0:
        return (0, 0, 0, 0)
    shells = _ECP_CORE_SHELLS.get(ecp_electrons // 2)
    if shells is None:
        return None
    counts = [0, 0, 0, 0]
    for _, l in shells:
        counts = [a + b for a, b in zip(counts, shell_irreps_c2v(l))]
    return tuple(counts)


def full_valence_space(atoms: Sequence[str], charge: int, *,
                       ecp_electrons: dict[str, int] | None = None) -> dict | None:
    """Full-valence CAS (ns np, d block ns (n-1)d), frozen cores inactive.

    Counts are all-electron (the SOC stage uses ANO-RCC without ECPs) unless
    ``ecp_electrons`` names the core electrons an ECP removes per element (PySCF
    molecules in def2 bases: ``pyscf_valence_space``).  None for open f shells or
    when the electron count does not fit the space.
    """
    inactive, ras2, valence_electrons, d_block = [0, 0, 0, 0], [0, 0, 0, 0], 0, False
    for el in atoms:
        z = _z(el)
        ncore = valence_core_orbitals(z)
        core = None if ncore is None else core_irreps_c2v(ncore)
        removed = ecp_core_irreps_c2v((ecp_electrons or {}).get(el, 0))
        if core is None or removed is None:
            return None
        core = tuple(a - b for a, b in zip(core, removed))
        d_block |= is_d_block(z)
        inactive = [a + b for a, b in zip(inactive, core)]
        ras2 = [a + b for a, b in zip(ras2, valence_shells_c2v(z))]
        valence_electrons += z - 2 * ncore
    nact = valence_electrons - charge
    if not 0 < nact < 2 * sum(ras2):
        return None
    return {"inactive": tuple(inactive), "ras2": tuple(ras2), "active_electrons": nact,
            "rule": "FULL_VALENCE_NS_NP_ND" if d_block else "FULL_VALENCE_NS_NP"}


# ------------------------------------------------------------------ basis + ECP

@dataclass(frozen=True)
class LoadedBasis:
    shells: Any
    ecp: Any            # PySCF ECP spec or None (all-electron)
    ecp_electrons: int  # core electrons replaced by the ECP
    provenance: str


def _split_nwchem(text: str) -> tuple[str, str]:
    head, sep, tail = text.partition("\nECP\n")
    return head, (sep + tail if sep else "")


@lru_cache(maxsize=None)
def load_basis(name: str, element: str) -> LoadedBasis | None:
    """Basis plus its ECP for one element: PySCF library first, then BSE.

    A basis designed for an ECP must never be used without it: PySCF loads e.g.
    def2-TZVPPD for Ag silently as all-electron (47 electrons in a basis built for
    19) when no ``ecp`` is passed.
    """
    from pyscf import gto
    try:
        shells = gto.basis.load(name.lower(), element)
        if shells:
            ecp = None
            try:
                ecp = gto.basis.load_ecp(name.lower(), element) or None
            except Exception:  # noqa: BLE001 - no ECP of that name in the library
                ecp = None
            if ecp is not None or _z(element) <= 36:
                return LoadedBasis(shells, ecp, int(ecp[0]) if ecp else 0, f"{name}(PYSCF)")
    except Exception:  # noqa: BLE001 - absent from the PySCF library
        pass
    try:
        import basis_set_exchange as bse
        text = bse.get_basis(name, elements=[element], fmt="nwchem", header=False)
    except Exception:  # noqa: BLE001 - absent from BSE as well
        return None
    head, ecp_text = _split_nwchem(text)
    shells = gto.basis.parse(head, symb=element)
    if not shells:
        return None
    ecp = gto.basis.parse_ecp(ecp_text, symb=element) if ecp_text else None
    return LoadedBasis(shells, ecp or None, int(ecp[0]) if ecp else 0, f"{name}(BSE_{bse.version()})")


def frozen_orbitals(element: str, ecp_electrons: int) -> int | None:
    """Frozen spatial orbitals of one atom beyond its ECP core (None: no rule)."""
    n = valence_core_orbitals(_z(element))
    if n is None or ecp_electrons % 2 or ecp_electrons // 2 > n:
        return None
    return n - ecp_electrons // 2


@lru_cache(maxsize=None)
def element_route(element: str) -> dict:
    """{"route", "reason", "ecp_electrons"} for one element (rules in the module docstring)."""
    z = _z(element)
    if z <= 36:
        return {"route": ROUTE_ALL_ELECTRON, "reason": "Z<=36", "ecp_electrons": 0}
    if valence_core_orbitals(z) is None:
        return {"route": None, "reason": "OPEN_F_SHELL_OR_BEYOND_RN", "ecp_electrons": None}
    loaded = [load_basis(PP_ROUTE_SERIES.format(x=CARDINAL_LETTER[x]), element) for x in PP_ROUTE_CARDINALS]
    if any(b is None or b.ecp is None for b in loaded):
        return {"route": None, "reason": f"NO_{PP_ROUTE_SERIES.format(x='X')}_SERIES", "ecp_electrons": None}
    nelec = {b.ecp_electrons for b in loaded}
    if len(nelec) != 1 or frozen_orbitals(element, loaded[0].ecp_electrons) is None:
        return {"route": None, "reason": "INCONSISTENT_ECP_CORE", "ecp_electrons": None}
    return {"route": ROUTE_PSEUDOPOTENTIAL, "reason": "PP_SERIES_AVAILABLE", "ecp_electrons": nelec.pop()}


def molecule_basis(atoms: Sequence[str], name_by_element: dict[str, str]) -> dict | None:
    """PySCF ``basis``/``ecp`` keyword arguments for the named basis per element.

    None if a basis is missing or a heavy element's basis has no ECP (fail closed).
    """
    basis, ecp, provenance = {}, {}, {}
    for el in dict.fromkeys(atoms):
        loaded = load_basis(name_by_element[el], el)
        if loaded is None or (_z(el) > 36 and loaded.ecp is None):
            return None
        basis[el] = loaded.shells
        provenance[el] = loaded.provenance
        if loaded.ecp is not None:
            ecp[el] = loaded.ecp
    return {"basis": basis, "ecp": ecp, "provenance": provenance}


# ------------------------------------------------------------ basis names + PySCF

def basis_name_for(element: str, family: str) -> str:
    """Element-specific member of a correlation-consistent family.

    PP elements take the -PP variant of an aug-/cc- family name (aug-cc-pVQZ ->
    aug-cc-pVQZ-PP, aug-cc-pwCVTZ -> aug-cc-pwCVTZ-PP); def2 and other names are
    returned unchanged (def2 carries its own ECPs).
    """
    if element_route(element)["route"] != ROUTE_PSEUDOPOTENTIAL:
        return family
    low = family.lower()
    if low.startswith(("aug-cc-p", "cc-p", "d-aug-cc-p")) and not low.endswith("-pp"):
        return family + "-PP"
    return family


@lru_cache(maxsize=None)
def resolve_named_basis(name: str, element: str) -> LoadedBasis | None:
    """``load_basis`` plus the d-aug construction (one more even-tempered diffuse
    primitive per angular momentum) for a ``d-`` name that no library has."""
    loaded = load_basis(name, element)
    if loaded is not None or not name.lower().startswith("d-"):
        return loaded
    base = load_basis(name[2:], element)
    if base is None:
        return None
    from .cbs_stage import _even_tempered_extra_diffuse
    return LoadedBasis(_even_tempered_extra_diffuse(base.shells), base.ecp, base.ecp_electrons,
                       f"d-{base.provenance}(EVEN_TEMPERED_CONSTRUCTED)")


def mole_basis_kwargs(atoms: Sequence[str], basis: Any) -> dict:
    """``basis``/``ecp`` keyword arguments for ``gto.M`` of a molecule of ``atoms``.

    Light-element molecules get ``{"basis": basis}`` unchanged.  A heavy element
    (Z > 36) must be given by basis *name* (one name for all atoms, or a per-element
    mapping); its shells and ECP come from ``resolve_named_basis`` (all-electron
    bases such as dyall-av4z or x2c-QZVPPall come without one).  Shells without a
    name or a missing basis raise ValueError (fail closed: PySCF would otherwise
    silently treat an ECP basis as all-electron).
    """
    heavy = [el for el in dict.fromkeys(atoms) if _z(el) > 36]
    if not heavy:
        return {"basis": basis}
    per_element = dict(basis) if isinstance(basis, dict) else {el: basis for el in dict.fromkeys(atoms)}
    ecp = {}
    for el in heavy:
        spec = per_element.get(el)
        if not isinstance(spec, str):
            raise ValueError(f"heavy element {el} needs a named basis to attach its ECP, got {type(spec).__name__}")
        loaded = resolve_named_basis(spec, el)
        if loaded is None:
            raise ValueError(f"basis {spec!r} not available for {el}")
        per_element[el] = loaded.shells
        if loaded.ecp is not None:  # BSE lists an ECP with every basis built for one
            ecp[el] = loaded.ecp
    return {"basis": per_element, "ecp": ecp}


def implicit_core_orbitals(mol: Any) -> int:
    """Core orbitals an ECP replaces for PP-route atoms that ``mol`` treats all-electron.

    The chain's correlation model for a PP-route element is "everything outside the
    ECP core"; an all-electron member (the sfX2C side of the ECP correction) keeps
    that core uncorrelated so both members correlate the same orbitals.  Zero for
    molecules of light elements or with the ECPs attached.
    """
    n = 0
    for i in range(mol.natm):
        el = mol.atom_pure_symbol(i)
        z = _z(el)
        if z > 36 and int(mol.atom_charge(i)) == z:
            route = element_route(el)
            if route["route"] == ROUTE_PSEUDOPOTENTIAL:
                n += route["ecp_electrons"] // 2
    return n


def named_series(element: str, cardinals: Sequence[int], family: str) -> dict[int, tuple[str, str]] | None:
    """{X: (basis name, provenance)} when the named family loads at every X.

    ``family`` takes ``{x}`` (cardinal letter D/T/Q/5) or ``{n}`` (digit, e.g.
    dyall-av{n}z).  A -PP family must come with its ECP.
    """
    out = {}
    for x in cardinals:
        name = family.format(x=CARDINAL_LETTER[x], n=x)
        loaded = load_basis(name, element)
        if loaded is None or (family.lower().endswith("-pp") and loaded.ecp is None):
            return None
        if loaded.ecp is not None and loaded.ecp_electrons != element_route(element)["ecp_electrons"]:
            return None  # another core size would change the correlated space
        out[x] = (name, loaded.provenance)
    return out


def pyscf_valence_space(atoms: Sequence[str], charge: int, basis: str) -> dict | None:
    """``full_valence_space`` for a PySCF molecule in the named ``basis`` (with its ECPs)."""
    ecp = {}
    for el in dict.fromkeys(atoms):
        if _z(el) > 36:
            loaded = load_basis(basis, el)
            if loaded is None:
                return None
            ecp[el] = loaded.ecp_electrons
    return full_valence_space(atoms, charge, ecp_electrons=ecp)
