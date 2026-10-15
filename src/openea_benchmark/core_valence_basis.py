"""Element-wise core-valence basis rule for the automated CV stage (Patch 46).

Delta_CV = EA(all-electron) - EA(frozen core) needs, for every atom that has
core electrons, a basis with tight core-correlating functions.  The rule is
per element and molecule-independent; it never looks at the molecule:

1. no core electrons (H, He): the valence family aug-cc-pVXZ;
2. otherwise the first available of
   aug-cc-pwCVXZ, cc-pwCVXZ, aug-cc-pCVXZ, cc-pCVXZ
   from PySCF's library, then from the local Basis Set Exchange package;
3. a family without "aug" gets the diffuse functions of aug-cc-pVXZ appended
   (the most diffuse primitive per angular momentum that aug-cc-pVXZ adds to
   cc-pVXZ), because anions need them;
4. nothing available: ``None`` - the stage fails closed for that molecule.

"Core" is the noble-gas core frozen in the frozen-core CBS
(``valence_frozen_core_orbitals``), so AE - FC correlates exactly the orbitals
the CBS layer left out.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

CARDINAL_LETTER = {2: "D", 3: "T", 4: "Q", 5: "5", 6: "6"}
CV_FAMILIES = ("aug-cc-pwCV{x}Z", "cc-pwCV{x}Z", "aug-cc-pCV{x}Z", "cc-pCV{x}Z")


@lru_cache(maxsize=None)
def core_orbital_count(element: str) -> int:
    """Frozen-core orbitals the CV layer correlates in addition (beyond an ECP core)."""
    from pyscf import gto
    from .heavy_elements import ROUTE_PSEUDOPOTENTIAL, element_route, frozen_orbitals
    route = element_route(element)
    if route["route"] == ROUTE_PSEUDOPOTENTIAL:
        return frozen_orbitals(element, route["ecp_electrons"])
    from .adaptive.stage3_execution import valence_frozen_core_orbitals
    z = gto.charge(element)
    mol = gto.M(atom=f"{element} 0 0 0", basis="sto-3g", spin=z % 2, verbose=0)
    return int(valence_frozen_core_orbitals(mol))


def _load(name: str, element: str) -> tuple[list, str] | None:
    """Shells of a named basis for one element, PySCF library first, then BSE."""
    from pyscf import gto
    try:
        shells = gto.basis.load(name.lower(), element)
        if shells:
            return shells, f"{name}(PYSCF)"
    except Exception:  # noqa: BLE001 - absent from the PySCF library
        pass
    try:
        import basis_set_exchange as bse
        text = bse.get_basis(name, elements=[element], fmt="nwchem", header=False)
        shells = gto.basis.parse(text, symb=element)
        if shells:
            return shells, f"{name}(BSE_{bse.version()})"
    except Exception:  # noqa: BLE001 - absent from BSE as well
        pass
    return None


def _single_primitive_exponents(shells: list) -> dict[int, list[float]]:
    by_l: dict[int, list[float]] = {}
    for shell in shells:
        l, prims = int(shell[0]), shell[1:]
        for column in range(1, len(prims[0])):
            nonzero = [p[0] for p in prims if abs(float(p[column])) > 0.0]
            if len(nonzero) == 1:
                by_l.setdefault(l, []).append(float(nonzero[0]))
    return by_l


def augmenting_diffuse_shells(element: str, cardinal: int) -> list | None:
    """The diffuse primitives aug-cc-pVXZ adds to cc-pVXZ (one per l)."""
    x = CARDINAL_LETTER[cardinal]
    aug, plain = _load(f"aug-cc-pV{x}Z", element), _load(f"cc-pV{x}Z", element)
    if aug is None or plain is None:
        return None
    base = {l: set(round(e, 10) for e in v) for l, v in _single_primitive_exponents(plain[0]).items()}
    extra = []
    for l, exps in sorted(_single_primitive_exponents(aug[0]).items()):
        new = [e for e in exps if round(e, 10) not in base.get(l, set())]
        if new:
            extra.append([l, [min(new), 1.0]])
    return extra


def _family_basis(family: str, element: str, cardinal: int) -> tuple[Any, str] | None:
    x = CARDINAL_LETTER[cardinal]
    loaded = _load(family.format(x=x), element)
    if loaded is None:
        return None
    shells, provenance = loaded
    if family.startswith("aug-"):
        return shells, provenance
    diffuse = augmenting_diffuse_shells(element, cardinal)
    if diffuse is None:
        return None
    return list(shells) + diffuse, provenance + f"+DIFFUSE_FROM_aug-cc-pV{x}Z"


def element_series(element: str, cardinals: tuple[int, ...], families: tuple[str, ...]
                   ) -> dict[int, tuple[Any, str]] | None:
    """First family available at every requested X: {X: (PySCF spec, provenance)}.

    The cardinal series of one element never mixes families.  A family whose
    name does not start with "aug-" gets the aug-cc-pVXZ diffuse primitives.
    """
    for family in families:
        series = {x: _family_basis(family, element, x) for x in cardinals}
        if all(v is not None for v in series.values()):
            return series
    return None


# Beyond Kr (pseudopotential route): basis names, resolved with their ECPs when the
# molecule is built.  CV: aug-cc-pwCVXZ-PP (outer-core correlating functions).
# SR: the ECP carries scalar relativity, so the "relativistic correction" becomes the
# ECP correction EA[sfX2C, all-electron dyall-avXz, ECP core frozen] - EA[ECP, aug-cc-pVXZ-PP]
# (heavy_elements.implicit_core_orbitals keeps the correlated orbitals identical).
PP_CV_FAMILY = "aug-cc-pwCV{x}Z-PP"
PP_SR_PAIR = ("dyall-av{n}z", "aug-cc-pV{x}Z-PP")


def _pp_route(element: str) -> bool:
    from .heavy_elements import ROUTE_PSEUDOPOTENTIAL, element_route
    return element_route(element)["route"] == ROUTE_PSEUDOPOTENTIAL


def cv_element_series(element: str, cardinals: tuple[int, ...]) -> dict[int, tuple[Any, str]] | None:
    """Core-valence series of one element (rule in the module docstring)."""
    if _pp_route(element):
        from .heavy_elements import named_series
        return named_series(element, cardinals, PP_CV_FAMILY)
    if core_orbital_count(element) == 0:
        series = element_series(element, cardinals, ("aug-cc-pV{x}Z",))
        return None if series is None else {x: (spec, prov + " NO_CORE_ELECTRONS")
                                            for x, (spec, prov) in series.items()}
    return element_series(element, cardinals, CV_FAMILIES)


# Scalar relativity (Patch 48/48b): the SFX2C1E member uses a Douglas-Kroll
# recontracted family, the NR member its nonrelativistic partner (same name without
# "-DK"); both families must exist at every cardinal.
SR_FAMILIES = ("aug-cc-pV{x}Z-DK", "cc-pV{x}Z-DK")


def sr_element_series(element: str, cardinals: tuple[int, ...]) -> dict[int, tuple[Any, str]] | None:
    """{X: ((DK spec, NR spec), provenance)} for the first family pair available at every X."""
    if _pp_route(element):
        from .heavy_elements import named_series
        rel, model = (named_series(element, cardinals, f) for f in PP_SR_PAIR)
        if rel is None or model is None:
            return None
        return {x: ((rel[x][0], model[x][0]), f"{rel[x][1]} | ECP {model[x][1]}") for x in cardinals}
    for family in SR_FAMILIES:
        rel = element_series(element, cardinals, (family,))
        nr = element_series(element, cardinals, (family.replace("-DK", ""),))
        if rel is not None and nr is not None:
            return {x: ((rel[x][0], nr[x][0]), f"{rel[x][1]} | NR {nr[x][1]}") for x in cardinals}
    return None
