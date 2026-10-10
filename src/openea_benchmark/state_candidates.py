"""Symmetry-labelled electronic-state candidates for a diatomic from two element symbols.

No tables of experimental atomic or molecular data are used:

1. Atomic configurations follow the Madelung (aufbau) order; for elements whose
   valence shell competes between ns and (n-1)d / np, single promotions are
   added as alternative configurations instead of guessing the lowest one.
2. Russell-Saunders terms (L, S, parity) of every configuration are generated
   by exact microstate counting.
3. Wigner-Witmer correlation: every pair of atomic terms on a dissociation
   channel yields molecular states (2S+1, Lambda, +/- for Sigma, g/u for like
   atoms).  The +/- and g/u labels are obtained from the explicit action of
   the reflection sigma_v and (for identical nuclei) the inversion on the
   product basis |m_A, m_B>; validated against Herzberg's correlation tables
   (OH, CN, O2, N2, F2 in tests).

Which candidate is the ground state is NOT predicted here: energies are a
computational question.  The output enumerates what must be covered.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations

# Madelung order (n, l) up to 7p; enough for the periodic table.
_MADELUNG = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2), (4, 1), (5, 0), (4, 2), (5, 1),
             (6, 0), (4, 3), (5, 2), (6, 1), (7, 0), (5, 3), (6, 2), (7, 1)]
_CAPACITY = {0: 2, 1: 6, 2: 10, 3: 14}
_L_LETTER = "SPDFGHIKLMNOQRTUV"
_LAMBDA_LETTER = {0: "Sigma", 1: "Pi", 2: "Delta", 3: "Phi", 4: "Gamma", 5: "H", 6: "I"}


def atomic_number(symbol: str) -> int:
    from pyscf.data.elements import charge
    return int(charge(symbol))


@dataclass(frozen=True, order=True)
class AtomicTerm:
    two_s: int      # 2S
    l: int          # total L
    parity: int     # +1 even, -1 odd

    @property
    def label(self) -> str:
        return f"{self.two_s + 1}{_L_LETTER[self.l]}{'' if self.parity > 0 else 'o'}"


@dataclass(frozen=True)
class AtomicState:
    element: str
    charge: int
    configuration: str
    term: AtomicTerm
    configuration_kind: str  # GROUND_AUFBAU | PROMOTION

    @property
    def label(self) -> str:
        sign = "" if self.charge == 0 else ("-" if self.charge == -1 else f"{self.charge:+d}")
        return f"{self.element}{sign}({self.term.label}; {self.configuration})"


@dataclass(frozen=True, order=True)
class MolecularSymmetry:
    two_s: int
    lam: int
    gu: str | None = None     # 'g'/'u' for like nuclei, else None
    reflection: str | None = None  # '+'/'-' for Sigma, else None

    @property
    def label(self) -> str:
        sym = _LAMBDA_LETTER.get(self.lam, f"L{self.lam}")
        return f"{self.two_s + 1}{sym}{self.gu or ''}{self.reflection or ''}"


@dataclass
class StateCandidate:
    charge: int
    symmetry: MolecularSymmetry
    count: int
    channels: list[str] = field(default_factory=list)
    single_determinant_high_spin: bool = False


# ----------------------------------------------------------------- atoms

def ground_configuration(z: int) -> list[tuple[int, int, int]]:
    """Aufbau occupation [(n, l, electrons), ...] for z electrons."""
    left, occ = z, []
    for n, l in _MADELUNG:
        if left <= 0:
            break
        k = min(left, _CAPACITY[l])
        occ.append((n, l, k))
        left -= k
    if left > 0:
        raise ValueError("electron count beyond supported Madelung order")
    return occ


def configuration_label(occ) -> str:
    return " ".join(f"{n}{'spdf'[l]}{k}" for n, l, k in occ if k)


def _valence_promotions(occ) -> list[list[tuple[int, int, int]]]:
    """Single promotions that commonly compete at low energy (ns->np, ns->(n-1)d, (n-1)d->ns)."""
    out = []
    shells = {(n, l): k for n, l, k in occ}
    outer_s = max((n for n, l, k in occ if l == 0 and k > 0), default=None)
    if outer_s is None:
        return out

    def moved(src, dst):
        new = dict(shells)
        if new.get(src, 0) <= 0 or new.get(dst, 0) >= _CAPACITY[dst[1]]:
            return None
        new[src] -= 1
        new[dst] = new.get(dst, 0) + 1
        return sorted([(n, l, k) for (n, l), k in new.items() if k > 0], key=lambda x: _MADELUNG.index((x[0], x[1])))

    candidates = [((outer_s, 0), (outer_s, 1))]
    if (outer_s - 1, 2) in shells or outer_s >= 4:
        candidates += [((outer_s, 0), (outer_s - 1, 2)), ((outer_s - 1, 2), (outer_s, 0))]
    for src, dst in candidates:
        if dst not in _MADELUNG or (dst[1] == 2 and outer_s < 4):
            continue  # e.g. no 1p shell for H/He
        cfg = moved(src, dst)
        if cfg is not None:
            out.append(cfg)
    return out


def terms_of_configuration(occ) -> list[AtomicTerm]:
    """Russell-Saunders terms by microstate counting over all open subshells."""
    open_shells = [(l, k) for n, l, k in occ if 0 < k < _CAPACITY[l]]
    parity = 1
    for n, l, k in occ:
        parity *= (-1) ** (l * k)
    tally: Counter = Counter({(0, 0): 1})
    for l, k in open_shells:
        spin_orbitals = [(ml, ms) for ml in range(-l, l + 1) for ms in (1, -1)]
        shell = Counter()
        for combo in combinations(spin_orbitals, k):
            shell[(sum(x[0] for x in combo), sum(x[1] for x in combo))] += 1
        new = Counter()
        for (a_ml, a_ms), na in tally.items():
            for (b_ml, b_ms), nb in shell.items():
                new[(a_ml + b_ml, a_ms + b_ms)] += na * nb
        tally = new
    terms = []
    while any(v > 0 for v in tally.values()):
        ml = max(k[0] for k, v in tally.items() if v > 0)
        ms = max(k[1] for k, v in tally.items() if v > 0 and k[0] == ml)
        terms.append(AtomicTerm(two_s=ms, l=ml, parity=parity))
        for m_l in range(-ml, ml + 1):
            for m_s in range(-ms, ms + 1, 2):
                tally[(m_l, m_s)] -= 1
    return sorted(set(terms), key=lambda t: (-t.two_s, -t.l))


def atomic_states(element: str, charge: int, *, include_promotions: bool = True) -> list[AtomicState]:
    z = atomic_number(element) - charge
    if z <= 0:
        return []
    configs = [("GROUND_AUFBAU", ground_configuration(z))]
    if include_promotions:
        configs += [("PROMOTION", c) for c in _valence_promotions(configs[0][1])]
    out = []
    for kind, occ in configs:
        label = configuration_label(occ)
        for term in terms_of_configuration(occ):
            out.append(AtomicState(element, charge, label, term, kind))
    return out


# ------------------------------------------------------------ Wigner-Witmer

def wigner_witmer(a: AtomicState, b: AtomicState, *, identical_nuclei: bool,
                  electrons_per_atom: int | None = None) -> list[MolecularSymmetry]:
    """Molecular symmetries (with multiplicity, one entry per state) from two atomic states."""
    la, lb = a.term.l, b.term.l
    sa, sb = Fraction(a.term.two_s, 2), Fraction(b.term.two_s, 2)
    like = identical_nuclei and a.element == b.element and a.charge == b.charge \
        and a.term == b.term and a.configuration == b.configuration
    # sigma_v |m_a, m_b> = s * |-m_a, -m_b> with s = w_a w_b (-1)^(L_a+L_b) (-1)^(m_a+m_b)
    s_refl = a.term.parity * b.term.parity * (-1) ** (la + lb)
    out = []
    spin = abs(sa - sb)
    while spin <= sa + sb:
        two_s = int(2 * spin)
        if like:
            # inversion = fermionic exchange of the two atoms' electron groups
            # (-1)^N_atom x spin-coupling exchange symmetry (-1)^(2 S_atom - S)
            # x orbital swap (m_a, m_b) -> (m_b, m_a)
            factor = (-1) ** electrons_per_atom * (-1) ** int(2 * sa - spin)
        for lam in range(0, la + lb + 1):
            basis = [(ma, lam - ma) for ma in range(-la, la + 1) if -lb <= lam - ma <= lb]
            if not basis:
                continue
            if lam > 0:
                if like:
                    # swap (m_a, m_b) -> (m_b, m_a): symmetric/antisymmetric combinations
                    seen, n_sym, n_anti = set(), 0, 0
                    for ma, mb in basis:
                        if (ma, mb) in seen:
                            continue
                        seen |= {(ma, mb), (mb, ma)}
                        if ma == mb:
                            n_sym += 1
                        else:
                            n_sym += 1
                            n_anti += 1
                    g_count = n_sym if factor > 0 else n_anti
                    u_count = n_anti if factor > 0 else n_sym
                    out += [MolecularSymmetry(two_s, lam, "g")] * g_count
                    out += [MolecularSymmetry(two_s, lam, "u")] * u_count
                elif identical_nuclei:
                    out += [MolecularSymmetry(two_s, lam, "g")] * len(basis)
                    out += [MolecularSymmetry(two_s, lam, "u")] * len(basis)
                else:
                    out += [MolecularSymmetry(two_s, lam)] * len(basis)
                continue
            # Lambda = 0: joint eigenstates of sigma_v (and swap for like atoms)
            seen = set()
            for ma, mb in basis:
                if (ma, mb) in seen:
                    continue
                partner = (-ma, -mb)
                seen |= {(ma, mb), partner}
                if partner == (ma, mb):  # (0,0)
                    refl = [s_refl]
                    swap = [1]
                else:  # pair -> symmetric (+s) and antisymmetric (-s); swap maps (m,-m)->(-m,m) too
                    refl = [s_refl, -s_refl]
                    swap = [1, -1]
                for r, sw in zip(refl, swap):
                    pm = "+" if r > 0 else "-"
                    if like:
                        gu = "g" if factor * sw > 0 else "u"
                        out.append(MolecularSymmetry(two_s, 0, gu, pm))
                    elif identical_nuclei:
                        out += [MolecularSymmetry(two_s, 0, "g", pm), MolecularSymmetry(two_s, 0, "u", pm)]
                    else:
                        out.append(MolecularSymmetry(two_s, 0, None, pm))
        spin += 1
    return out


# ------------------------------------------------------------------ channels

def dissociation_channels(atom_a: str, atom_b: str, charge: int) -> list[tuple[int, int, str]]:
    """(charge_a, charge_b, kind) channels for a molecule of total charge 0 or -1."""
    if charge == 0:
        return [(0, 0, "NEUTRAL_ATOMS"), (1, -1, "ION_PAIR"), (-1, 1, "ION_PAIR")]
    if charge == -1:
        return [(-1, 0, "ANION_ATOM"), (0, -1, "ANION_ATOM")]
    raise ValueError("only charge 0 and -1 are supported")


def enumerate_state_candidates(atom_a: str, atom_b: str, charge: int, *,
                               include_promotions: bool = True,
                               include_ion_pairs: bool = True) -> list[StateCandidate]:
    identical = atom_a == atom_b
    merged: dict[MolecularSymmetry, StateCandidate] = {}
    for qa, qb, kind in dissociation_channels(atom_a, atom_b, charge):
        if kind == "ION_PAIR" and not include_ion_pairs:
            continue
        states_a = atomic_states(atom_a, qa, include_promotions=include_promotions)
        states_b = atomic_states(atom_b, qb, include_promotions=include_promotions)
        for sa in states_a:
            for sb in states_b:
                if identical and qa == qb and (sb.term, sb.configuration) < (sa.term, sa.configuration):
                    continue  # unordered pair for like atoms
                syms = wigner_witmer(sa, sb, identical_nuclei=identical,
                                     electrons_per_atom=atomic_number(atom_a) - qa)
                channel = f"{sa.label} + {sb.label}"
                for sym, n in Counter(syms).items():
                    cand = merged.setdefault(sym, StateCandidate(charge, sym, 0))
                    cand.count += n
                    if channel not in cand.channels:
                        cand.channels.append(channel)
    return sorted(merged.values(), key=lambda c: (c.symmetry.two_s, c.symmetry.lam, c.symmetry.label))


def candidate_spin_sectors(atom_a: str, atom_b: str, charge: int, **kw) -> list[int]:
    """All 2S values that occur among the candidates (replaces hand-written spin lists)."""
    return sorted({c.symmetry.two_s for c in enumerate_state_candidates(atom_a, atom_b, charge, **kw)})
