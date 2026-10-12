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
    from pyscf import gto
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


def cv_element_series(element: str, cardinals: tuple[int, ...]) -> dict[int, tuple[Any, str]] | None:
    """One basis family for all requested cardinals: {X: (PySCF spec, provenance)}.

    The first family (in ``CV_FAMILIES`` order) available at every requested X
    is used, so the cardinal series of one element never mixes families.
    """
    families = ("aug-cc-pV{x}Z",) if core_orbital_count(element) == 0 else CV_FAMILIES
    for family in families:
        series = {x: _family_basis(family, element, x) for x in cardinals}
        if all(v is not None for v in series.values()):
            if core_orbital_count(element) == 0:
                series = {x: (spec, prov + " NO_CORE_ELECTRONS") for x, (spec, prov) in series.items()}
            return series
    return None
