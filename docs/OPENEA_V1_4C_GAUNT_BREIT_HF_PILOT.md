# OpenEA v1 — Four-component Dirac–HF Gaunt/Breit pilot (Patch 26)

## Objective and strict physical scope

PySCF 2.14 exposes an actual four-component Dirac–Hartree–Fock (DHF)
calculation with Dirac–Coulomb, Dirac–Coulomb–Gaunt and
Dirac–Coulomb–Breit Hamiltonians. This patch executes a *matched* trio
on a single explicitly authorized neutral or anion state and separately
computes their mean-field total-energy differences.

**This is NOT the residual spin-free two-electron correction to the existing
SFX2C1E/CCSD(T) EA.** The four-component Hamiltonian mixes scalar and
spin-dependent physics, and the Gaunt/Breit interactions overlap conceptually
with spin–orbit physics. Nor do mean-field differences constitute a
correlated CCSD(T) correction or a systematic, certified accuracy bound.
**Do not add them to the separately computed FCI-SISO SOC term.**

The code refuses to promote diagnostic differences to a production EA
component. It does not set the missing relativistic G3 correction to zero.

## What exactly is calculated

For each charge state q and fixed geometry/basis, perform distinct SCFs:

- E_DC(q): 4c Dirac–Coulomb (`with_gaunt=False`, `with_breit=False`)
- E_DCG(q): Dirac–Coulomb–Gaunt (`with_gaunt=True`, `with_breit=False`)
- E_DCB(q): Dirac–Coulomb–Breit (`with_gaunt=False`, `with_breit=True`)

In PySCF, **`with_breit=True` already includes the Gaunt interaction**;
never sum the DCG and DCB shifts as independent energy terms.

For each state, report E_DCG−E_DC, E_DCB−E_DC, and E_DCB−E_DCG.
For the neutral/anion pair, report differences of these differences as
*diagnostic EA sensitivities*, using Delta(EA)=Delta(E_neutral)-Delta(E_anion).

The pilot explicitly uses a point-charge nuclear model and the entire
small-small Coulomb integral block (`with_ssss=True`), with the same
all-electron orbital basis, spin input, nuclear geometry and numerical
convergence settings across the three Hamiltonians. The two charge states
may use different equilibrium geometries but must use the same basis policy.
The spin input alone is NOT a proof of a unique four-component electronic
state; the state identity remains an independent scientific review question.

Results are `COMPLETE_REVIEW_REQUIRED`, `POLICY_BLOCKED` or `ERROR`;
`COMPLETE_REVIEW_REQUIRED` **never** means `PRODUCTION_SOC_CLEARED` or
`SCALAR_RELATIVITY_REMAINDER_CLEARED`.

## Relation to the missing X2C two-electron correction

A fully matched scalar two-electron picture-change study requires a
validated spin-free reduction/decoupling and matching of the mean-field and
correlated electronic Hamiltonians. 4c DHF DC vs DCG/DCB alone does not supply
this and does not decompose the Gaunt/Breit result into scalar and SOC parts.

Consequently these three quantities are **not additive** to the existing
NR-vs-SFX2C1E CCSD(T) EA or CASSCF/CASCI FCI-SISO SOC correction. The
unquantified 2e X2C picture-change remainder is still an OPEN G3 component.

## Real integration smoke

```bash
cd /srv/storage/homes/analysis/dschmid/photodetachment/openea-benchmark
source .venv/bin/activate
nice -n 19 python scripts/openea_four_component_dhf_smoke.py --neutral-only
nice -n 19 python scripts/openea_four_component_dhf_smoke.py
```

These are respectively 3 and 6 genuine 4c-DHF HF calculations using a tiny
STO-3G LiH reference example. They establish **software integration only**;
STO-3G, point nuclei and HF do not yield a reliable high-accuracy EA.
Kramers/symmetry, numerical stability and state identity should be inspected
before extending to any heavier system. No external dependency is required.

Source references: PySCF Quickstart, `scf/05-breit_gaunt.py`, `pyscf.scf.dhf`
API, and PySCF X2C spin-free 1e documentation. For formal 2e picture-change
see the DIRAC xamfX2C documentation. Code must follow the installed PySCF
2.14.0 API; the genuine Artemis smoke is authoritative.
