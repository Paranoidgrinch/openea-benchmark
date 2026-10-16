"""Element rules beyond Kr: pseudopotential route (heavy-element step 1).

Every rule here is per element and molecule-independent.

Route (``element_route``):

* ALL_ELECTRON - Z <= 36, the existing all-electron chain (frozen core, aug-cc-pVXZ,
  CV in aug-cc-pwCVXZ, scalar relativity as sfX2C(DK basis) - NR).
* PSEUDOPOTENTIAL - Z > 36 when the correlation-consistent PP series
  aug-cc-pV{T,Q,5}Z-PP loads for the element (PySCF library, else Basis Set
  Exchange).  The ECP replaces the inner core and carries scalar relativity; the
  def2 bases of the scout/MR stages come with their own def2 ECPs.
* ALL_ELECTRON_X2C - Z > 36 without a PP series (Rb, Sr, Cs, Ba, La-Lu, Ac-Lr):
  all electrons, spin-free X2C Hamiltonian for the whole molecule in every stage
  (``apply_hamiltonian``), X2C-contracted cc-pVXZ-X2C / cc-pwCVXZ-X2C bases (D, T, Q;
  aug- where BSE has it, else the even-tempered "e-aug-" extension).  Scalar
  relativity is then part of the model, not a correction.
* None - fail closed with a reason: no basis series for the element.

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
ROUTE_X2C = "ALL_ELECTRON_X2C"
F_BLOCK = frozenset(range(57, 72)) | frozenset(range(89, 104))
X2C_ROUTE_SERIES = "cc-pV{x}Z-X2C"
X2C_ROUTE_CARDINALS = (2, 3, 4)
PP_ROUTE_SERIES = "aug-cc-pV{x}Z-PP"
PP_ROUTE_CARDINALS = (3, 4, 5)
CARDINAL_LETTER = {2: "D", 3: "T", 4: "Q", 5: "5", 6: "6"}

# Core shells in the order they are frozen; cumulative counts give the core sizes.
_CORE_SHELLS = ((1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2),
                (5, 0), (5, 1), (4, 3), (5, 2), (6, 0), (6, 1))
# (last Z of the range, frozen spatial orbitals).  Lanthanides [Kr]4d10 (5s5p correlated
# with 4f 5d 6s), actinides [Xe]4f14 5d10 (6s6p correlated with 5f 6d 7s).
_FROZEN_CORE_BY_RANGE = ((2, 0), (10, 1), (18, 5), (30, 9), (36, 14), (48, 18), (54, 23),
                         (56, 27), (71, 23), (80, 34), (86, 39), (88, 43), (103, 39))
# Valence shells of the full-valence CAS, by C2v irrep (a1, b1, b2, a2).
VALENCE_S, VALENCE_SP, VALENCE_SD = (1, 0, 0, 0), (2, 1, 1, 0), (3, 1, 1, 1)


def _z(element: str) -> int:
    from pyscf import gto
    return int(gto.charge(element))


def valence_core_orbitals(z: int) -> int | None:
    """All-electron spatial orbitals of the valence frozen core (None beyond Lr)."""
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


VALENCE_SDF = (5, 3, 3, 2)  # ns (n-1)d (n-2)f


def valence_shells_c2v(z: int) -> tuple[int, int, int, int]:
    if z <= 2:
        return VALENCE_S
    if z in F_BLOCK:
        return VALENCE_SDF
    return VALENCE_SD if is_d_block(z) else VALENCE_SP


def cas_core_orbitals(z: int) -> int | None:
    """Inactive orbitals of the full-valence CAS: the frozen core, plus for the f block
    the outer (n-1)s(n-1)p shells that CC correlates but the CAS keeps doubly occupied."""
    n = valence_core_orbitals(z)
    return None if n is None else n + (4 if z in F_BLOCK else 0)


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
        ncore = cas_core_orbitals(z)
        core = None if ncore is None else core_irreps_c2v(ncore)
        removed = ecp_core_irreps_c2v((ecp_electrons or {}).get(el, 0))
        if core is None or removed is None:
            return None
        core = tuple(a - b for a, b in zip(core, removed))
        d_block |= is_d_block(z) or z in F_BLOCK
        inactive = [a + b for a, b in zip(inactive, core)]
        ras2 = [a + b for a, b in zip(ras2, valence_shells_c2v(z))]
        valence_electrons += z - 2 * ncore
    nact = valence_electrons - charge
    if not 0 < nact < 2 * sum(ras2):
        return None
    if determinant_count(sum(ras2), nact) > MAX_CAS_DETERMINANTS:
        return None  # beyond conventional CASSCF; fail closed in the MR layers
    return {"inactive": tuple(inactive), "ras2": tuple(ras2), "active_electrons": nact,
            "rule": "FULL_VALENCE_NS_NP_ND" if d_block else "FULL_VALENCE_NS_NP"}


# Low-spin determinant count above which the full-valence CAS is not attempted
# (f-element molecules: 13 valence orbitals per f atom).
MAX_CAS_DETERMINANTS = 5.0e7


def determinant_count(norb: int, nelec: int) -> float:
    from math import comb
    na = (nelec + 1) // 2
    return float(comb(norb, na) * comb(norb, nelec - na))


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
        return {"route": None, "reason": "BEYOND_LR", "ecp_electrons": None}
    loaded = [load_basis(PP_ROUTE_SERIES.format(x=CARDINAL_LETTER[x]), element) for x in PP_ROUTE_CARDINALS]
    if any(b is None or b.ecp is None for b in loaded):
        x2c = [load_basis(X2C_ROUTE_SERIES.format(x=CARDINAL_LETTER[x]), element) for x in X2C_ROUTE_CARDINALS]
        if all(b is not None and b.ecp is None for b in x2c):
            return {"route": ROUTE_X2C, "reason": "NO_PP_SERIES_X2C_SERIES_AVAILABLE", "ecp_electrons": 0}
        return {"route": None, "reason": "NO_PP_OR_X2C_SERIES", "ecp_electrons": None}
    nelec = {b.ecp_electrons for b in loaded}
    if len(nelec) != 1 or frozen_orbitals(element, loaded[0].ecp_electrons) is None:
        return {"route": None, "reason": "INCONSISTENT_ECP_CORE", "ecp_electrons": None}
    return {"route": ROUTE_PSEUDOPOTENTIAL, "reason": "PP_SERIES_AVAILABLE", "ecp_electrons": nelec.pop()}


# ------------------------------------------------------------ basis names + PySCF

_DEF2_CARDINAL = {"def2-svpd": "D", "def2-tzvppd": "T", "def2-qzvppd": "Q",
                  "def2-svp": "D", "def2-tzvpp": "T", "def2-qzvpp": "Q"}


def _x2c_member(element: str, family: str) -> str:
    """X2C-route member of a family name: aug-cc-pVQZ -> aug-cc-pVQZ-X2C (or the
    even-tempered e-aug-cc-pVQZ-X2C where BSE has no aug- set), def2-XZVPPD -> the
    augmented X2C set of the same cardinal."""
    low = family.lower()
    if low.endswith(("-x2c", "-dk3")):
        return family
    if low in _DEF2_CARDINAL:
        family = f"aug-cc-pV{_DEF2_CARDINAL[low]}Z"
        low = family.lower()
    prefix = ""
    if low.startswith("d-"):
        prefix, family, low = "d-", family[2:], low[2:]
    if not low.startswith(("aug-cc-p", "cc-p")):
        return prefix + family
    name = family + "-X2C"
    if low.startswith("aug-") and load_basis(name, element) is None:
        name = "e-" + name  # even-tempered augmentation of cc-pVXZ-X2C
    return prefix + name


def basis_name_for(element: str, family: str) -> str:
    """Element-specific member of a basis family name.

    PP elements take the -PP variant of an aug-/cc- family name (aug-cc-pVQZ ->
    aug-cc-pVQZ-PP); def2 keeps its own ECPs.  X2C-route elements take the
    X2C-contracted all-electron member (``_x2c_member``).  Light elements: unchanged.
    """
    route = element_route(element)["route"]
    if route == ROUTE_X2C:
        return _x2c_member(element, family)
    if route != ROUTE_PSEUDOPOTENTIAL:
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
    low = name.lower()
    if loaded is not None or not low.startswith(("d-", "e-aug-")):
        return loaded
    # d-NAME: NAME plus one even-tempered diffuse primitive per l; e-aug-cc-X: the
    # same extension of cc-X (no published aug- set for the element).
    base = resolve_named_basis(name[2:] if low.startswith("d-") else name[len("e-aug-"):], element)
    if base is None:
        return None
    from .cbs_stage import _even_tempered_extra_diffuse
    tag = "d-" if low.startswith("d-") else "e-aug-"
    return LoadedBasis(_even_tempered_extra_diffuse(base.shells), base.ecp, base.ecp_electrons,
                       f"{tag}{base.provenance}(EVEN_TEMPERED_CONSTRUCTED)")


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
        if element_route(el)["route"] == ROUTE_X2C:
            spec = _x2c_member(el, spec)  # generic stage names (def2-TZVPPD, aug-cc-pVQZ) -> X2C set
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
            elif route["route"] == ROUTE_X2C:
                n += deep_core_orbitals(z)
    return n


def deep_core_orbitals(z: int) -> int:
    """X2C-route atoms: the shells a small-core ECP would replace, never correlated
    (all-electron members correlate the outer core above them): Rb, Sr [Ar]3d10;
    Cs, Ba [Kr]4d10; lanthanides [Ar]3d10; actinides [Kr]4d10 4f14."""
    if z in F_BLOCK:
        return 14 if z <= 71 else 30
    return 14 if z <= 38 else 23


def named_series(element: str, cardinals: Sequence[int], family: str) -> dict[int, tuple[str, str]] | None:
    """{X: (basis name, provenance)} when the named family loads at every X.

    ``family`` takes ``{x}`` (cardinal letter D/T/Q/5) or ``{n}`` (digit, e.g.
    dyall-av{n}z).  A -PP family must come with its ECP.
    """
    out = {}
    for x in cardinals:
        name = family.format(x=CARDINAL_LETTER[x], n=x)
        loaded = resolve_named_basis(name, element)
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


# ------------------------------------------------------------------ Hamiltonian

def requires_x2c(atoms: Sequence[str]) -> bool:
    """True when an X2C-route element is present: the whole molecule then uses sfX2C."""
    return any(_z(el) > 36 and element_route(el)["route"] == ROUTE_X2C for el in dict.fromkeys(atoms))


MODEL_HAMILTONIAN_ENV = "OPENEA_MODEL_HAMILTONIAN"


def set_model_hamiltonian(atoms: Sequence[str]) -> str:
    """Fix the run's Hamiltonian from the molecule (inherited by point-pool workers).

    Fragments must use the molecule's Hamiltonian: an O atom computed
    nonrelativistically next to an sfX2C LaO would shift the asymptote by O's
    relativistic energy (~1 eV).
    """
    import os
    value = "SFX2C1E" if requires_x2c(atoms) else "NONE"
    os.environ[MODEL_HAMILTONIAN_ENV] = value
    return value


def apply_hamiltonian(mf: Any) -> Any:
    """``mf`` with the chain's Hamiltonian: spin-free X2C when the run's molecule
    contains an X2C-route element (``set_model_hamiltonian``) or this system does
    (idempotent); unchanged otherwise."""
    import os
    mol = mf.mol
    if hasattr(mf, "with_x2c"):
        return mf
    if os.environ.get(MODEL_HAMILTONIAN_ENV) == "SFX2C1E" or requires_x2c(
            [mol.atom_pure_symbol(i) for i in range(mol.natm)]):
        return mf.sfx2c1e()
    return mf
