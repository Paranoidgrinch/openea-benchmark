# OpenEA v1 — planned method inventory after G2 simplification

**Basis:** `OPENEA_V1_ARCHITECTURE_AND_HANDOVER` canonical production
prescription (sections 3–19, 21, 24) and the repository's current
`OPENEA_V1_CLOSURE_GRAPH_AUDIT` / `OPENEA_V1_PRODUCTION_EXECUTION_ROUTING`.
This is an implementation inventory, NOT a statement that each method must
be evaluated for every molecule.

## Implemented executable foundations (not yet universal end-to-end validated)

| Planned role | Existing code/status | Important qualification |
|---|---|---|
| State candidate generation / Stage-2 reconnaissance | Infrastructure present | G1 completeness remains a distinct reviewed scientific question; heuristics cannot silently establish ground-state identity |
| State following, Stage-3 CCSD(T) PEC | Generic runner/identity framework | A calculation running successfully does not certify all competing states/asymptotes |
| Cardinal and diffuse refinement | Generic adaptive runners | Arbitrary CBS-model repair and electronic geometry/PEC transfer closure remain gaps |
| Core-valence correction | Matched AE/FC runner | Depends on valid all-electron/core-valence basis policy and convergence |
| One-electron scalar relativity | Matched NR/SFX2C1E runner | Does not include all two-electron relativistic/picture-change terms |
| CCSDT vs CCSD(T) | Generic **diagnostic** runner | Conditional, authorized; not required for every anion; no automatic CCSDTQ |
| J=0 anharmonic nuclear motion | Generic radial solver + PEC refinement/model comparison | Not DBOC/nonadiabatic remainder; nuclear isotope/state/PEC quality must be checked |
| Reference gate / MR branch contract | Present; MR branch fail-closed | No universal validated MR production solver |
| G3a–G3e, error budget, final BOUND/UNBOUND/UNRESOLVED engine | Contracts and bridge present | An incomplete physical correction or method leaves relevant gates open |
| G2 localized valence path | Typed multidimensional evidence, no mandatory EOM | Actual automated density/localization/reference evidence adapter not yet connected; scientific review required |
| Optional EOM/stabilization/continuum diagnostics | Existing modules retained | Not universal default; CAP executable backend deliberately not required |

## Remaining planned scientific method/correction work

1. **SOC relevance assessment and (if necessary) validated SOC correction**.
   Need generic method/capability selection, electronic-state/spin-manifold
   identities and uncertainty for *both* neutral and anion. FeH-specific
   SOC pilot is NOT a universal production runner. SOC is not allowed to
   disappear as `0±0`; it can be bounded below target tolerance with evidence.
2. **Multireference production branch.** The existing branch interface only
   stops unsafe SR calculations. Need select/validate an open-source MR
   energy treatment, consistent neutral/anion active-space policy, state/PEC
   continuity, basis treatment, correlation/dynamic effects and uncertainties.
   This is essential to claim coverage of diatomics with serious MR character,
   but not mandatory for systems that pass the SR reference gate.
3. **Residual scalar relativity** beyond SFX2C1E: assess/bound missing
   two-electron terms, picture-change and related Hamiltonian effects for
   chemically relevant regimes. Choose a matched benchmark/correction
   route and avoid double counting with SOC/CV.
4. **DBOC / nonadiabatic nuclear remainder**, and isotope-dependent
   error assessment when needed. The J=0 BO vibrational solve is present,
   but cannot by itself certify full adiabatic nuclear physics. Terms may
   be physically bounded without high-cost calculations when justified.
5. **CBS model / fixed-geometry/PEC-transfer closure**: adapt the existing
   energy and Stage-3 machinery to handle unresolved extrapolation model
   sensitivity and transfer of corrections from limited bond lengths to the
   neutral/anion equilibrium PECs, retaining uncertainties and state identity.
   These are missing generic adapters, not wholly new quantum methods.
6. **Post-CCSD(T) model adequacy closure**: CCSDT is an available diagnostic.
   Need robust small/stable vs large/unstable handling, potential justified
   DeltaT3 transfer/error bound, feedback into G3a/G3c and precision planner.
   **No** automatic CCSDTQ/FCI escalation. Higher-order CC remains
   validation research only.
7. **G1/state-completeness and G2 evidence automation**: make the electronic
   manifold discovery, neutral/anion identity, asymptote inventory, orbital
   density/localization, SCF stability and direct detachment estimates feed
   the reviewed gates reproducibly. The current G2 valence object is a
   contract, not a new density/scf executable. Difficult attachment
   characters may need optional EOM or genuine continuum treatment; no
   single one-size-fits-all diagnostic should be forced.
8. **Full G3 error-budget closure/orchestration**: preserve additive EA0
   physics and non-overlapping corrections, including uncertainty from
   model, PEC/geometry, SOC, SR remainder, nuclear remainder and MR branch.
   The contracts exist but arbitrary open components do not yet all have
   automated evidence-producing pathways.

## Development order and validation discipline

Suggested next implementation: **SOC relevance + correction strategy**, then
**MR production capability** (for the broad diatomic mission), followed by
scalar two-electron and beyond-BO remainder policies/runners and CBS transfer
closure. The two ordering choices may be revisited based on the target
validation set; do not launch all methods for all systems. In parallel,
connect G1/G2 evidence and uncertainty to executable data.

Only after the necessary correction/review paths exist, run a **tiered**
scientific validation campaign: basic SR valence anions -> weak/diffuse cases
-> heavy/SOC-sensitive species -> genuine MR diatomics. A preliminary
software integration smoke is useful earlier but is NOT a completed
scientific universal end-to-end validation.

## Explicitly outside the default graph

- Mandatory EA-EOM, CAP, UKRmol+/scattering or automated broad continuum scans.
- CCSDTQ/CCSDTQP/FCI as automatic rescue after CCSDT instability.
- Pretending that omitted SOC, DBOC, 2-electron SR, MR or basis model
  uncertainties equal zero.

### Phase C Patch 21 — Conditional generic SOC pilot

A backend-executing *development* runner (`adaptive/soc_fci_siso_runner.py`)
now generalizes the FeH CASSCF/CASCI/AMFI FCI-SISO state-interaction procedure.
Its numerical SOC shifts are **CASCI-based diagnostic candidates** and do not
close item 1 (SOC production method + uncertainty). Scientific spin manifold,
active-space, basis, relativistic Hamiltonian, energy-model and geometry
convergence remain explicitly outstanding. The method is invoked only with
manual authorization, not part of every diatomic default workflow.

### Phase C Patch 23 — matched-geometry numerical CBS/PEC transfer

`adaptive/cbs_pec_transfer.py` now evaluates **already computed and
state-reviewed** high-minus-low electronic model differences over finite bond
length grids, then interpolates at the *separate* neutral/anion target
geometries. It accepts actual completed Stage-3 CCSD(T) point energies and
refuses unmatched states, model swaps, provenance reuse, missing data and
any target outside the computed interval. A PySCF HF-only smoke verifies
multi-geometry numerical wiring, not the scientific CBS result.

This closes a piece of the missing *numerical* transfer functionality but
NOT model-bound/CBS convergence, automatic high-level grid scheduling,
or final G3 uncertainty certification. Remaining two-electron scalar
relativistic/picture-change and beyond-BO DBOC/nonadiabatic remainders
are still explicitly open; no missing correction is set to zero.


### Phase C Patch 24 — HF-only numerical DBOC pilot

`adaptive/dboc_finite_difference.py` performs an **actual finite-difference
Hartree–Fock Slater-determinant DBOC** on two distinct nuclear step sizes,
using displaced PySCF RHF/ROHF/UHF SCFs, cross-AO overlaps, explicitly sourced
isotopic nuclear masses, and all six Cartesian nuclear coordinates.
This is a method-development diagnostic, not a validated high-level DBOC.
Correlated DBOC, translational/center-of-mass convention benchmarking,
nonadiabatic couplings and the nuclear remainder **remain open**.
The HF diagnostic may not automatically clear any G3 gate.

### Phase C Patch 26 — explicit 4c-DHF Gaunt/Breit sensitivity (diagnostic only)

`adaptive/four_component_two_electron_probe.py` computes the matched
Dirac–Coulomb/Gaunt/Breit *mean-field* operator sensitivity on fixed neutral
and anion geometries with explicit authorization and real PySCF DHF backends.
This **does not implement** the missing *spin-free* two-electron X2C
picture-change remainder, is not CCSD(T) correlated, and cannot be summed
with FCI-SISO SOC without double counting. The physical G3 remainder stays
OPEN; high-accuracy relativistic benchmarking remains necessary.
