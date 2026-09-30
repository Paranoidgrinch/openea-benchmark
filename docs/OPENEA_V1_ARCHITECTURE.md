# OpenEA v1 scientific architecture (draft 0.1)

Date: 2026-09-30. Status: **development contract, NOT frozen production policy**.

## Scientific objective

Input: initially a diatomic formula with optional isotope specification. Output:
(1) positive adiabatic electron affinity `EA0` with a supported uncertainty;
(2) `UNBOUND` when sufficient theoretical evidence excludes a stable 0 K anion
ground state / nonpositive EA; (3) `UNRESOLVED` when either decision is unsupported.
For `UNBOUND`, do not waste computation resolving a precise negative EA. Distinguish
attachment stability, nuclear vibrational binding, and thermodynamic adiabatic EA.
The final EA0 compares the *lowest v=0 levels* of relevant neutral and anion
states, with consistent relativity and SOC when required. Ensure state identity
before taking a minimum.

## Non-negotiable invariants

1. The production dependency graph has no access to experimental EA data. Validate
   only after an immutable theoretical prediction has been recorded.
2. Chemical rules propose risks; none pre-decide EA sign or eliminate states.
3. SCF convergence != reference stability != state identity != physical binding.
4. No mean of DFT predictions as final EA and no averaging of methods to fabricate
   a statistical uncertainty.
5. Unknown corrections do not equal zero. Non-convergence does not equal UNBOUND.
6. Estimated ranges are not claimed to be rigorous mathematical error bounds or
   automatically 95% confidence intervals. Their evidence provenance is recorded.
7. Never add scalar-relativistic, SOC or other corrections twice to a baseline
   that already contains the same physical effect.
8. Minimize redundant work via immutable state, result and checkpoint provenance.

## Stage boundaries

`ChemicalPrior` (no molecule calculations): composition, electron parity, atomic
valence possibilities, open d/f shells, polarization clues, heavy-element alerts,
symmetry and fragment-channel hypotheses. No known experimental reference values.

`ElectronicReconnaissance`: multi-guess, spin candidates, stability tests with
scope, invariant identity fingerprints, physical manifolds, branch continuity,
initial neutral/anion PEC, first diffuse test. Reuse existing OpenEA modules.

`DiagnosticResolution`: D01–D12 (listed below), invoked as relevant and re-openable
when later evidence conflicts with an earlier assumption. Diagnostic results
are NOT methods; each must record its inputs, scientific meaning and limitations.

`AdaptiveProduction`: choose coherent SR, MR, attachment/continuum, state-resolution
and physical-corrections routes based on diagnosed risks and remaining uncertainty.

`Decision`: require separately reviewed G1/G2/G3. Return `BOUND`, `UNBOUND`, or
`UNRESOLVED`; for positive EA, separately report precision target status.

## The twelve diagnostics

| ID | Question | Triggers and evidence | Escalation |
|---|---|---|---|
| D01 | Spin representation | S, S², reference alternatives | Alternative or spin-adapted reference |
| D02 | SCF stability | Internal/external scope, unstable modes | Reoptimize unstable solution |
| D03 | Static correlation | Multiple indicators, NO universal T1 cutoff | CASSCF; CAS expansion; MR branch |
| D04 | Competing states | Candidate energies, tracked PEC and error ranges | Refine plausible competitors |
| D05 | Reference sensitivity | Compare truly distinct reference routes | Alternate reference or MR |
| D06 | Perturbative triples reliability | (T), amplitudes, selected CCSDT checks | CCSDT, higher correlation, revisit D03 |
| D07 | Additional diffusity | Augmentation vs cardinal basis convergence separately | More diffuse test + conditioning checks |
| D08 | Physical attachment | VDE, density, EOM-EA if helpful, continuum sensitivity | Attachment/continuum resolution |
| D09 | PEC/asymptotes | Root continuity, minima, dissociation channels | Sample PEC adaptively |
| D10 | Scalar relativity | Hamiltonian accounting, EA sensitivity | Compatible X2C/DKH |
| D11 | SOC | Term content, SOC EA contribution, state interaction | OpenMolcas RASSI/SOC candidate route |
| D12 | Nuclear motion | Anharmonic v=0 and PEC convergence | Refined PEC / radial solve |

Passing a diagnostic requires concrete evidence; an unavailable test is NOT a
successful test. `NOT_APPLICABLE` needs a stated physical rationale.

## Method selection contracts

SR starts provisionally if state/reference quality permits CCSD diagnostics. Only
then may a consistent CCSD(T) baseline be produced. Check separately cardinal
basis convergence, extra diffusity, higher-order triples, core-valence, scalar
relativity, SOC, and nuclear motion. Anomalies can reopen earlier diagnostics.

MR begins after an evidential need, not just the presence of a transition metal.
Build an orbital union relevant to both charge states; use AVAS/APC or suitable
natural orbitals, retain symmetry-related groups, validate CAS enlargement and
state continuity. CASSCF/NEVPT2 and OpenMolcas CASSCF/CASPT2/RASSI are *distinct*
validated-software candidates, not a universal ranked accuracy ladder. Track
CASPT2 intruders/variant/level shift and state averaging if used.

Attachment route checks whether a finite-basis anion is bound instead of a
continuum discretization; EOM-EA is a useful complementary method but is itself
a finite-basis electronic treatment. A vertical attachment diagnostic never
silently replaces a fully relaxed EA0.

## Scientific decision record

- `G1`: sufficiently complete neutral/anion state search for the actual verdict.
- `G2`: attachment/continuum and relevant nuclear binding adequately resolved.
- `G3`: uncertainty model adequate for *sign decision*, not necessarily target
  numerical precision.
- `ea0_interval_ev` may be absent when an unbounded/unknown correction remains.
- `critical_open_questions` always block sign certification.
- With all gates independently `CLEARED`: interval lower > 0 => `BOUND`;
  upper <= 0 => `UNBOUND`; otherwise `UNRESOLVED`.
- With any open gate: `UNRESOLVED` irrespective of apparent numerical sign.
- For `BOUND`, interval half-width <= target => `TARGET_MET` else
  `TARGET_NOT_MET`. These are model-derived, not automatically calibrated errors.

## Error budget

Treat a consistent neutral-minus-anion difference as the primary object. Each
correction has explicit (possibly asymmetric) residual bounds and scientific
provenance: numerical, geometry, basis cardinality, diffusity, post-(T), core-
valence, scalar relativity, SOC, vibrational, mixed/cross terms and state
selection. Correlated errors must not be naively treated as independent.

For included candidate states with energy intervals [L_i,U_i], the min-envelope
is [min_i L_i, min_i U_i]. If N and A are such envelopes, a conservative EA
range is [L_N - U_A, U_N - L_A]. This operation does not establish missing
state coverage or cancel correlated errors automatically.

## Validation and initial acceptance set

Use synthetic evidence first: identical entangled pi-orientations, missing ROHF
external stability, erroneous/ambiguous roots, a large T1 without decisive MR
proof, a small T1 but other bad diagnostics, active space instability, finite-
basis false binding, sign-changing uncertainty, unknown SOC, and simulated
resource exhaustion. Then verify transfers on OH, CN, MgH, AlO, N2 and FeH.
Existing experimental reference values are not accessible in production.

## Repo compatibility observations (public GitHub, 2026-09-30)

- `src/openea_benchmark/root_record.py` has `FAILED`, `CONVERGED`,
  `CANONICALIZED` SCF statuses, correctly not EA claims.
- `state_identity.py` already distinguishes SAME/DISTINCT/AMBIGUOUS and requires
  explicit validated thresholds; do not duplicate or weaken it.
- `provenance/CAPABILITY_GATE_001.md`: PySCF/CCpy higher-order CC capability.
- `CAPABILITY_GATE_002.md`: validated OpenMolcas 26.06 software baseline.
- `CAPABILITY_GATE_003.md`: molecule-level OH CASSCF/CASPT2/RASSI-SOC chain.
- Public README reports provisional `r2SCAN`, `r2SCANh`, `PBE0` scout panel,
  `ma-def2-TZVPP`, `aug-pcseg-2` cross-check, `def2-QZVPPD` anchor. Do NOT
  restart scout method selection or assume these are frozen production policy.
- The public README and the user's local commit description may represent
  different development points; compare Git hashes before editing existing code.

Public references:
https://github.com/Paranoidgrinch/openea-benchmark
https://raw.githubusercontent.com/Paranoidgrinch/openea-benchmark/main/docs/WORKFLOW_SCOPE.md
https://raw.githubusercontent.com/Paranoidgrinch/openea-benchmark/main/src/openea_benchmark/root_record.py
https://raw.githubusercontent.com/Paranoidgrinch/openea-benchmark/main/src/openea_benchmark/state_identity.py
