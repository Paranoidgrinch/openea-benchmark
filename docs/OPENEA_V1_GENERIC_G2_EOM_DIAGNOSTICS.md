# OpenEA v1 — G2 EA-EOM raw diagnostics and controlled diffuse scaling (Patch 12)

## Scope and scientific status

Patch 12 executes explicitly authorized *neutral-reference* EA-EOM-CCSD diagnostics using PySCF: closed-shell RHF/RCCSD and open-shell UHF/UCCSD. These are additional electron-attachment calculations for a validated single-reference **neutral** state; they are **not** interchangeable with an anion-reference energy, do not by themselves establish the target anion multiplicity, and do not prove that a finite Gaussian basis has excluded continuum pseudostates.

The runner produces all requested **raw** EA-EOM roots, with energies expressed both as PySCF charged-excitation eigenvalues, `E(N+1)-E(N)`, and OpenEA's `EA = -omega`. **Every raw root remains `ROOT_REVIEW_REQUIRED`.** The EOM solver cannot infer across-basis root identity merely from the ordinal root index, eigenvalue, or finite-basis convergence. The current module deliberately does not fabricate a one-particle/Dyson weight. The CCSD/EOM method family is related to the ΔCC energy family; treat it as complementary electronic-structure evidence, not independent experimental confirmation.

## Input contract

`run_g2_eom_diagnostics` needs:

- a state-identity- and reference-character-validated neutral, with geometry, charge, spin, state ID, source-root ID and an actual PySCF-readable SCF source checkpoint;
- an explicit `basis_by_element` for every augmentation level in one declared basis family, sorted and unique;
- per-subpoint authorization (`aug:0`, `aug:1`, etc.), with rationale and provenance;
- optional stabilizations declared by `scale:<augmentation>:<factor>` and exact reviewed `DiffuseShellSelector(element, shell_index)` selections;
- a checkpoint directory for robust resumability, and concrete EOM root/convergence settings.

The default runtime calls PySCF `RCCSD` or `UCCSD` followed by `eomea_method().kernel(nroots=..., koopmans=False)`. The neutral reference is seeded from the *source* checkpoint with projection to the current orbital basis; SCF stability and CCSD/EOM convergence are checked. No molecule-specific charge/spin, frozen core, or basis mappings are guessed. Only all-electron one-electron basis policies are accepted in this version; MR/ECP and other references need their own validated paths.

## Diffuse stabilization

The optional exponent-scaling helper modifies **only explicitly identified simple uncontracted single-primitive Gaussian shells** after loading the element-resolved PySCF basis. It never guesses which exponent is 'most diffuse', silently scales contractions, or changes unspecified elements. Unsupported contracted shells, invalid indices, and missing elements fail before any expensive job. A scan must use multiple authorized scale points and independent continuum/root review downstream.

Energy stationarity under a short exponent-scaling scan is **not** a rigorous continuum exclusion theorem. In particular, a stable pseudostate or avoided crossing can appear deceptively stationary. At the next layer, the G2 scientific review must consider root continuity, spectral avoided crossings, spin/symmetry selection, attachment localization and convergence against enlargement of the one-electron space. Ambiguous or near-threshold spectra remain `UNRESOLVED`.

## Checkpoints and read-only recovery

One JSON record is stored per raw G2 EOM subpoint. Its SHA-256 signature includes the state, entire basis policy **after** scaling, explicit backend/PySCF version, solver settings, and SHA-256 of the neutral source checkpoint. Compatible completed records may be reused without new authorization; **new** computations require exact subpoint authorization. Invalid checkpoints raise a mismatch error rather than being silently overwritten. Writes are atomic.

`evidence_points_from_review` converts a selected raw root to Patch-11 `EOMEAAttachmentPoint` or `StabilizationPoint`, **only with a separately supplied `Review`**. It never creates a `CLEARED` review itself; a CLEARED review must cite independent root-identity evidence. A default unresolved review cannot close D08.

## Explicit limitations and next work

This patch implements a real EOM backend and controlled basis stabilization **execution** (to be smoke-tested on Artemis); it does **not** implement automatic cross-basis Dyson orbital overlap/root tracking, classify resonances from avoided crossings, or automate independent continuum reviews. Those remain the next G2 scientific tasks. There is no legal path for a raw EOM result to produce a terminal `BOUND`/`UNBOUND` alone.

The standalone `scripts/openea_g2_eom_runner_smoke.py` runs a cheap **LiH/STO-3G software integration smoke**, not a valid LiH electron affinity. It should report `COMPLETE_ROOT_REVIEW_REQUIRED` and verify the checkpoint resume.
