# OpenEA Phase C — Patch 28: automatic axial-state equivalence on Stage-3 checkpoints

## Purpose

The first autonomous OH benchmark (`benchmark-auto` / Patch 27) completed all 35
high-level CCSD(T) points. The initialization-identity gates remained unresolved
for neutral OH (doublet) and OH-anion (triplet) because separate ROHF starts had
strongly different density fingerprints, even though CCSD(T) energies were almost
identical. **Matching energies alone does not prove the same state.**

Patch 28 adds a *symmetry-covariant*, generic diatomic electronic **pilot** review.
It is not an OH-specific exception, does not read experimental references, and
calculates no new CCSD(T) point on reanalysis.

## Physical test

For a diatomic molecule, rotations about its internuclear axis are physical
SO(2) symmetries of the spin-free electronic Hamiltonian. The code:

1. Loads the already-audited high-level HF checkpoints through the existing
   Stage-3 identity logic (no rerun).
2. Compares the existing spin-resolved density fingerprints as before.
3. Only on a non-cleared comparison, checks context consistency, same geometry,
   near-identical **HF** energies and `<S^2>` for initializations.
4. Constructs proper 3D rotations about the *actual internuclear axis*.
   **PySCF `mol.ao_rotation_matrix`** supplies the AO angular-momentum
   representation. An explicit `U† S U ≈ S` verification rejects invalid
   transformations, and Cartesian-AO cases fail closed.
5. Maximizes the minimum singular value of **both** spin-resolved occupied-space
   overlap matrices under one *common* axial angle. Different occupied-orbital
   gauges within a spin subspace do not affect the decision.
6. Only if all required alpha and beta singular values reach the explicit
   **pilot** threshold, records `EQUIVALENT` with the angle, actual singular
   values and AO metric residual. Otherwise `NOT_EQUIVALENT` or `UNAVAILABLE`.
7. Applies the same rotation-aware comparison to *adjacent PEC geometries*
   using genuine cross-geometry AO overlap integrals, retaining existing
   independent `BranchThresholds` and step-size/spin checks.

The decision only allows the *spin-free high-level PEC review* to progress when
all required safeguards pass. It does **not** prove the entire spin-orbit
manifold, identify the ground state, validate the final EA or remove open
high-spin competing PECs. Spin-orbit state interaction still needs full state
coverage. Matched orientation components are kept in recorded evidence.

## Stored evidence / automatic status

`OH/benchmark_auto/stage3_pec_reviews.json` now contains
`axial_symmetry_evidence` per Stage-3 job with comparison stage and request IDs,
angle, occupied-space singular values, metric residual and reason. The main
`auto_state.json` has `stage3_identity_summary` for every job (initialization and
geometry-continuity status, symmetry comparisons, local minimum status).
The normal CLI prints the same concise summary without an extra manual parser.

### Reanalyse *without calculating any new point*

After Patch 27 was committed and Patch 28 applied on Artemis, use the existing
reference-free OH manifest and directory:

```bash
cd /srv/storage/homes/analysis/dschmid/photodetachment/openea-benchmark
source .venv/bin/activate
nice -n 19 python -m openea_benchmark.workflow --atoms O H \
    --output ../openea_two_atom_v2 --auto-dry-run --threads 4 --max-stage3-points 0
```

This command reads cached Stage-3 point records, recomputes **only** identity,
continuity and guarded PEC classifications, and writes the updated report. If
any point is absent, it stays explicitly `PENDING`, not silently calculated.

## Scientific and operational limitations

- **Pilot thresholds, not validated production policy.** The default minimum
  rotated occupied-space singular value is 0.999; HF energy, spin and AO-metric
  tolerances are explicit settings, not silently fit to OH reference EA.
- Rotation is restricted to spin-free non-Cartesian GTO diatomics; non-axially
  related solutions and invalid AO transformations remain unresolved.
- Energetic degeneracy without actual orbital rotation equivalence never passes.
- HF rotational equivalence does not prove equality of correlated CCSD(T)
  energies; the existing independent Stage-3 duplicate-energy test still applies.
- The existing `WEEKEND_PILOT_NOT_PRODUCTION` label, `outcome=UNRESOLVED`,
  `ea_computed=false`, and `ground_state_validated=false` are retained.
- **Actual PySCF 2.14 / archived OH checkpoint replay must run on Artemis.**
  The local tests used synthetic orbital manifolds and an import-only PySCF stub;
  no local chemical checkpoint was silently declared validated.

## Next autonomous work after this checkpoint gate

Route the now-guarded high-level PEC into automatic equilibrium/nuclear-motion
and basis-convergence evidence; treat open asymptotic/high-spin competition
without inventing lower bounds; only then assemble the EA/error budget and
fail closed with `UNRESOLVED` where experimental-blind evidence is insufficient.
