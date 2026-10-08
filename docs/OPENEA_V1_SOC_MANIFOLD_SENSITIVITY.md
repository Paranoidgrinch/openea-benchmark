# OpenEA Phase C Patch 22 — Real SOC spin-manifold sensitivity

**Status: software/method validation diagnostic only. No final SOC correction or EA.**

The generic FCI-SISO runner (Patch 21) now works on Artemis for both LiH and
LiH−. Its output depends on the set of spin-free CASCI states admitted to the
spin-orbit Hamiltonian. This patch implements an actual, independently
authorized **restricted vs expanded state-manifold calculation** at an identical
geometry, basis, scalar Hamiltonian, active space, HF checkpoint and CASSCF
orbital-optimization protocol.

## Scientific calculation

For fixed nuclear charge state and geometry, execute two full PySCF/FCI-SISO
calculations with the *same* CASSCF reference spin and number of reference-spin
roots, but a strictly larger set of included spin-free CASCI states in the
second run. This isolates the numerical **included-state-manifold sensitivity**
more cleanly than comparing arbitrary CAS, geometries or state-average weights.

The observed difference is

\[
\delta^{\mathrm{SOC}}_{\rm manifold}
=\left[E_{\rm SOC,ground}-E_{\rm SF,ground}\right]_{\rm expanded}
-\left[E_{\rm SOC,ground}-E_{\rm SF,ground}\right]_{\rm restricted}.
\]

`observed_shift_change_ev` is reported for inspection; even an exactly zero
value does **not** bound the contribution from higher, uncomputed states,
missing correlation, basis changes, orbital/active-space choices, relativistic
model or final neutral/anion geometry. For EA the physical SOC sign remains
`δEA_SOC = δE_SOC(neutral) − δE_SOC(anion)`; the manifold test operates on each
charge separately and makes no final EA.

### Fail-closed comparison

- Both jobs must be independently cost-authorized; no implicit extra SOC run.
- Physical MRPointRequest, review IDs, checkpoint content digest, FCI-SISO source
  SHA256 and scalar Hamiltonian must match.
- Ground reference spin and number of ground-spin CASSCF averaging roots cannot
  change; expansion cannot remove previously included roots.
- Both results must contain all requested spin-free roots and spin projections.
- Retained spin-free CASCI root energies must agree within an explicitly small
  *numerical* drift threshold; energy ordering within each spin sector is only a
  fingerprint and does not validate root identity.
- Results are `CANDIDATE_REVIEW_REQUIRED` or `UNRESOLVED`; they **cannot** set
  `production_soc_validated`, `soc_uncertainty_bounded`, `spin_manifold_complete`,
  or `spin_free_state_identity_cleared`.

This module is separate from OpenEA's standard valence-ion electron-affinity
workflow. It runs only when SOC investigation is scientifically relevant.

## Artemis integration smoke

Requires the *existing* pinned `fci-siso` checkout from Patch 21; no new
external dependencies:

```bash
cd /srv/storage/homes/analysis/dschmid/photodetachment/openea-benchmark
source .venv/bin/activate
SISO_DIR=/srv/storage/homes/analysis/dschmid/photodetachment/openea_external/fci-siso
nice -n 19 python scripts/openea_soc_manifold_sensitivity_smoke.py --fci-siso-checkout "$SISO_DIR" --neutral-only
nice -n 19 python scripts/openea_soc_manifold_sensitivity_smoke.py --fci-siso-checkout "$SISO_DIR"
```

LiH/STO-3G is a code-integration smoke, not a physical SOC benchmark.
When scientifically validating SOC, use independent cases with appreciable
nonzero splittings, spin-manifold/basis/active-space studies, and correlation
consistent neutral/anion corrections. Only after that consider a method-level
SOC error bound; do not take the numerical sensitivity itself as the bound.

Other previously planned methods remain open: 2e scalar relativity/picture
change, DBOC/nonadiabatic remainder, CBS/PEC transfer and full G1–G3 closure.
