# Design notes

One note per layer of the `symmetry-auto` route (see the pipeline table in the top-level README).

| Layer | Notes |
|---|---|
| Scope | `WORKFLOW_SCOPE.md` |
| 1 Symmetry scout, state candidates, heavy elements | `OPENEA_V1_STATE_CANDIDATES.md` |
| 2 Fragment asymptotes / open curves | `OPENEA_V1_OPEN_PEC_EXTENSION.md`, `OPENEA_V1_ATTACHMENT_ASYMPTOTE.md`, `OPENEA_V1_FRAGMENT_SCF_RETRY.md`, `OPENEA_V1_HMINUS_TWOELECTRON_POLICY.md`, `OPENEA_V1_NUMPY24_NEWTON_COMPAT.md` |
| 3 Stage-3 CCSD(T) curves | `OPENEA_V1_BENCHMARK_AUTO_ACQUISITION.md`, `OPENEA_V1_PEC_SELECTION_BRIDGE.md`, `OPENEA_V1_ADAPTIVE_PLANNER.md`, `OPENEA_V1_STAGE3_*.md`, `OPENEA_V1_AUTOMATED_AXIAL_STATE_EQUIVALENCE.md`, `OPENEA_V1_CORRELATION_SPACE_CORRECTION.md` |
| 4 Reference branches, 6 CBS, 12 ZPE | `OPENEA_V1_REFERENCE_BRANCHES_AND_CBS.md`, `OPENEA_V1_COMPONENT_RESOLVED_CBS.md` |
| 5 Electronic EA | `OPENEA_V1_ELECTRONIC_EA_STAGE.md`, `OPENEA_V1_EQUILIBRIUM_RESOLVER.md`, `OPENEA_V1_ATTACHMENT_WORKFLOW.md`, `OPENEA_V1_ATTACHMENT_DECISION.md` |
| 7 Core-valence | `OPENEA_V1_FROZEN_CORE_CBS_AND_CORE_VALENCE.md`, `OPENEA_V1_GENERIC_CORE_VALENCE_RUNNER.md` |
| 8 Scalar relativity | `OPENEA_V1_SCALAR_RELATIVITY.md`, `OPENEA_V1_GENERIC_SCALAR_RELATIVITY_RUNNER.md` |
| 9 Spin–orbit | `OPENEA_V1_SPIN_ORBIT_STAGE.md` |
| 10 Post-CCSD(T) | `OPENEA_V1_POST_CCSD_T_STAGE.md` |
| 11 Multireference | `OPENEA_V1_MR_STAGE.md` |
| 12 Nuclear motion | `OPENEA_V1_GENERIC_NUCLEAR_MOTION.md`, `OPENEA_V1_NUCLEAR_MOTION_ORCHESTRATION.md` |

Notes on removed layers (old DFT-scout pipeline, G2/EOM continuum, DBOC, 4c probes,
production-evidence planning) are in tag `archive/pre-cleanup-2026-10-16`.
