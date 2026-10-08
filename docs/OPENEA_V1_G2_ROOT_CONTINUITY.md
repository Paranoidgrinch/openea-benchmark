# OpenEA v1 — G2 root-continuity candidates (Patch 13)

## Scientific status and objective

Patch 12 stores **raw** neutral-reference EA-EOM-CCSD roots, without across-basis identity. Patch 13 adds a reproducible *candidate-matching diagnostic* using the projected one-particle component (`r1`) of each **right** EA-EOM eigenvector in the AO basis. No root is automatically certified; `RootContinuityReport.review.status` is always `UNRESOLVED` and its status is at best `CANDIDATE_REVIEW_REQUIRED`.

**Not a Dyson orbital.** The CCSD right-EOM `r1` coefficients alone are not the interacting Dyson orbital or a many-body spectral pole strength. A proper Dyson observable needs appropriate left/right coupled-cluster density transition amplitudes. The stored algebraic `one_particle_amplitude_fraction` is a numerical diagnostic of the supplied right-EOM vector only; it is not a physical quasiparticle weight. It may be particularly misleading for strongly correlated or mixed 2p1h attachment roots. It is therefore never emitted as G2's `one_particle_weight`.

**Not continuum exclusion.** High overlaps and smooth/stationary finite-basis EOM eigenvalues do not prove the electronic state is square-integrable or bound below the continuum. Particularly for near-threshold/diffuse states, independent reviewed stabilization/continuum evidence remains mandatory. No G2 decision is made here, and `evidence_points_from_review` forbids promoting a `G2_ROOT_PROPOSAL:` ID alone into a CLEARED root-identity review.

## Explicit inputs

- A **complete** `G2EOMSeries` from the Patch-12 runner, with matching neutral source state/checkpoint hash, same electronic reference family and same basis family.
- An **explicit** `starting_root_index` on the lowest augmentation-level spectrum. The default is not assumed to be root zero or the most positive EA.
- A numeric `RootContinuitySettings` policy (`min_overlap`, `min_competitor_separation`, `min_algebraic_one_particle_fraction`). These are screening thresholds for *candidate generation*, not universal chemical thresholds or calibrated confidence intervals.
- AO overlap matrices for each pair. The default provider evaluates `pyscf.gto.intor_cross('int1e_ovlp', mol_a, mol_b)` on the **same atomic coordinates**, reconstructing each orbitally resolved basis (including explicitly authorized exponent scaling). A mock overlap provider is available for deterministic policy tests.

## Algorithm

For a root with AO-projected right-EOM amplitudes `c^(a)` and `c^(b)` (the latter absent in the RHF spin-adapted representation), evaluate

```
O_AB = |sum_spin c_A^T S_AB c_B| /
       sqrt[(sum_spin c_A^T S_AA c_A)(sum_spin c_B^T S_BB c_B)]
```

All three AO overlaps must have consistent dimensions and a physical metric. For real nonrelativistic calculations the phase of a root is arbitrary; the modulus removes the sign ambiguity. Mixed spin representations, non-finite values, non-positive AO norms and normalized overlaps greater than one are rejected.

Adjacent augmentation levels are matched by the **largest AO overlap**, not by matching ordinal root indices or electronic energies. Candidates must pass overlap, minimum 1p amplitude content, runner-up separation **and reciprocal source-root uniqueness** tests. A failed comparison cannot be silently bridged. Exponent-scaled stabilization points are compared separately to **their fixed augmentation-level baseline**, not chained through unrelated scaling levels.

The report records the proposed root mapping, link scores and reasons, source fingerprints and a SHA-256 `G2_ROOT_PROPOSAL:` ID. Its review is *always* UNRESOLVED. This may guide independent state-character and continuum review; only separate scientific evidence can eventually permit a CLEARED Review in the Patch-12 evidence bridge.

## Provenance and checkpoint compatibility

Patch 13 adds optional per-root `one_particle_ao_alpha`, `one_particle_ao_beta` and `one_particle_amplitude_fraction` to the existing JSON checkpoint payload. Imported Patch-12 root payloads remain readable, but without AO fingerprint information they cannot be root-tracked. The PySCF backend identifier changes to `...-v2-AO1P`, so attempting to reuse an incompatible old cache under the same filename **fails closed** (the previous cache is never silently interpreted as new evidence). To compute new fingerprint information, use a fresh checkpoint directory or deliberately archive the old cache and start a new authenticated run.

The new projection handles RHF/RCCSD and UHF/UCCSD separately and respects PySCF CCSD frozen masks. It is still an implementation-level backend integration that requires an actual PySCF Artemis software smoke before commit; the local unit suite uses injected overlap providers, not a substitute for PySCF.

## Remaining work

1. Run and verify the updated real PySCF smoke on Artemis (RHF; a separate UHF smoke should follow when a suitable convergent small reference is established).
2. Introduce independent root character diagnostics, left/right CCSD Dyson transition densities or comparable validated state-observable fingerprints, and external spin-sector/multiplicity reviews.
3. Develop a continuum-exclusion route for near-threshold systems, with explicit stabilization-vs-pseudostate discrimination. **Do not** convert a smooth EOM pseudo-continuum eigenvalue into boundness.
4. Wire reviewed results into the D08 `AttachmentContinuumAssessment` without weakening G1/G2/G3 fail-closed contracts.

## Downstream stabilization diagnostic (Patch 14)

`analyze_g2_stabilization_profile()` uses this proposal to select the raw
scaled-root energies and quantify exponent-scaling trends. It cannot approve
root identity, electron-boundness or continuum exclusion. A separate
independent continuum-discrimination review is needed before the D08
stabilization contract can resolve to STABLE_BOUND/STABLE_UNBOUND.
