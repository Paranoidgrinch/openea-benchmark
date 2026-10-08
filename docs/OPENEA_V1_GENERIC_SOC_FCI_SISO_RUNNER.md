# OpenEA v1 — Generic, conditional MR state-interaction SOC pilot (Patch 21)

**Scientific status: DIAGNOSTIC_METHOD_DEVELOPMENT. No production SOC, no final EA.**

## Purpose, scope and scientific distinction

The universal OpenEA diatomic workflow needs **conditional** SOC assessment, not
SOC calculation for every molecule. The pre-existing FeH study
`pilots/feh_high_accuracy/06_soc/run_soc.py` was a valuable method pilot but
hardcoded FeH orbitals, root manifolds and paths. This generic runner removes
those system-specific choices. The actual electronic-structure method is:

1. Reoptimize the supplied, checkpoint-seeded RHF/ROHF reference at *one* explicitly
   declared diatomic geometry and electronic Hamiltonian (`NONE` or `SFX2C1E`).
2. Optimize a shared active-orbital basis by **ground-spin state-averaged
   CASSCF**. The reference spin, orbital indices, active electrons, and roots
   are explicit—not guessed.
3. Solve **mixed-spin multi-root CASCI** in that *same* orbital basis, using
   explicitly selected spin multiplicities. Extract actual CI wavefunctions,
   spins and spin-free **CASCI** energies.
4. Construct a physically normalized state-derived AO density (doubly occupied
   inactive MOs plus active-space CI density) for the spin-orbit mean field.
5. Call pinned `FCISISO.kernel_we(dmao=..., amfi=True)` for AMFI
   state-interaction spin-orbit splitting (spin-projected spectrum).
6. Report the *diagnostic* spin-orbit shift for the lowest supplied spin-free
   versus lowest coupled energy. For neutral/anion only, propose
   `ΔEA_SOC = ΔE_SOC(neutral) − ΔE_SOC(anion)` in eV.

**Crucial limitation:** the coupling diagonal uses spin-free **CASCI** energies.
The preceding MR runner's SC-NEVPT2 totals are *not* diagonal SOC Hamiltonian
inputs here. Therefore this is not a NEVPT2+SOC total-energy model, not a
validated spin-orbit contribution to final adiabatic EA and not an uncertainty
bound. A changed CAS, basis, spin manifold, or different equilibrium geometry
can change the reported shift. Root identity across levels/spins and basis/model
convergence need independent review. Missing SOC terms cannot be set to `0±0`.

## Capability and reproducibility policy

- Uses the existing **third-party** open-source GPLv3 repository
  <https://github.com/hczhai/fci-siso>, pinned to exactly
  `e0f103104f45d10881d8a3526e1f654559142141` (as recorded in the FeH
  pilot). Its code is **not vendored** into OpenEA.
- The checkout is passed explicitly through `SOCPointSettings.fci_siso_checkout`.
  The runner checks the exact Git revision, tracked dirty status and computes
  the imported `fcisiso.py` SHA256. It does not silently install anything.
- User authorization constrains **each** CAS determinant sector and number of
  spin-orbit projected states. Missing checkout/budget/SCF failures are blocked
  or unresolved; backend errors do not create fake zeros or successful results.
- Results include request signatures and source checkpoint SHA256, but this
  first version does **not** implement a persisted SOC checkpoint.
- ECP calculations are not admitted. AMFI and scalar-relativistic
  **picture-change/2-electron** SOC model validation are outstanding.
- `SOCPointRequest` deliberately reuses the `MRPointRequest` atom, geometry,
  basis, charge, initial root, active-space and review-provenance fields.
- All computed energy values and pair shifts are `*_REVIEW_REQUIRED`; neither
  `SOCPointResult` nor `SOCEACandidate` can certify SOC validity or production.

## Installation / verification on Artemis

Keep this checkout **separate from the FeH pilot**, so neither its pinned
revision nor its working state can be changed by OpenEA development.
Explicitly clone it outside the OpenEA repository when absent:

```bash
cd /srv/storage/homes/analysis/dschmid/photodetachment/openea-benchmark
source .venv/bin/activate

SISO_DIR=/srv/storage/homes/analysis/dschmid/photodetachment/openea_external/fci-siso
mkdir -p "$(dirname "$SISO_DIR")"
if [ ! -d "$SISO_DIR/.git" ]; then
    git clone https://github.com/hczhai/fci-siso.git "$SISO_DIR"
fi
# First inspect any local modifications; checkout deliberately not forced.
git -C "$SISO_DIR" status --short
git -C "$SISO_DIR" rev-parse HEAD
# Only with a clean checkout:
if [ -z "$(git -C "$SISO_DIR" status --porcelain --untracked-files=no)" ]; then
    git -C "$SISO_DIR" fetch origin e0f103104f45d10881d8a3526e1f654559142141 &&
    git -C "$SISO_DIR" checkout --detach e0f103104f45d10881d8a3526e1f654559142141
else
    echo 'STOP: Existing FCI-SISO checkout contains tracked modifications'
fi
```

Smoke (run **lowest CPU scheduling priority**; start with neutral):

```bash
cd /srv/storage/homes/analysis/dschmid/photodetachment/openea-benchmark
source .venv/bin/activate
SISO_DIR=/srv/storage/homes/analysis/dschmid/photodetachment/openea_external/fci-siso
nice -n 19 python scripts/openea_soc_fci_siso_smoke.py \
    --fci-siso-checkout "$SISO_DIR" --neutral-only
nice -n 19 python scripts/openea_soc_fci_siso_smoke.py \
    --fci-siso-checkout "$SISO_DIR"
```

The smoke runs **LiH/STO-3G neutral** (CAS(2,2), singlet + triplet) and
**LiH⁻/STO-3G** (CAS(3,3), doublet + quartet) at 1.60 Å. All selections are
small *software-only demonstrations*, **not** scientifically validated
reference active spaces or electron affinities. It must report `status: PASS`,
`production_soc_validated: false`, `soc_uncertainty_bounded: false`.

## What remains after the first runtime smoke

- An **independent SOC benchmark** and an explicit spin-manifold convergence
  series, including potentially omitted nearby roots and competing spins.
- Validation of the one-electron/AMFI SOC operator, scalar Hamiltonian and
  picture-change consistency, and any double counting with relativistic terms.
- Spin-free energy-model compatibility with NEVPT2/CCSD(T), and a defensible
  model-error bound when transferring CASCI SOC shifts.
- Neutral/anion geometry/PEC and basis transfer, producing a reviewed and
  uncertainty-bounded correction for the final OpenEA EA resolver.

The initial bridge therefore stops at **candidate diagnostics**. This is an
actual conditional calculation path—not a new mandatory method or an
unfounded closure contract.
