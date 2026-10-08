# OpenEA — generic MR CASSCF / CASCI / strongly contracted NEVPT2 pilot

## Scope and method role

This is the **first real MR computation backend**, not an auto-approved high-accuracy electron affinity. It uses PySCF 2.14, already pinned in `pyproject.toml`. The runner does not install new packages, automatically choose active orbitals, identify the ground state, or authorize an EA.

For each explicitly requested diatomic geometry and charge/spin manifold, the runner executes:

1. Load the declared HF source checkpoint, reconverge RHF (singlet) or ROHF (open-shell) at the declared basis and geometry. No fallback to uncontrolled initial guess.
2. Select **explicit 0-based HF MO indices** in an explicit CAS(n electrons, m orbitals); enforce spin and determinant budget. No molecular name hardcodes.
3. Optimize CASSCF orbitals; for multiple roots, use equal-weight **state-average** over one prescribed spin manifold. Restrict this first executable pilot to one spin target; mixed-spin state averaging belongs to a separately validated extension.
4. Perform a **new multi-root CASCI** on those optimized orbitals. PySCF documents this as necessary for state-specific NEVPT2 after SA-CASSCF: [official PySCF example](https://github.com/pyscf/pyscf/blob/master/examples/mrpt/41-for_state_average.py).
5. Evaluate **strongly contracted NEVPT2** for every requested CASCI root. Store CASCI, correlation, total and spin expectation per root.

## Scientific restrictions

- Explicit high-cost authorization, explicit basis-by-element, state-manifold and active-space review IDs, existing checkpoint and finite determinant budget are required.
- Result status is `COMPLETE_REVIEW_REQUIRED`, never `VALIDATED_AVAILABLE`; the existing fail-closed `MRProductionCapability` / MR routing remains unchanged.
- Neither CASCI root number nor orbital index guarantees **state continuity** across R, basis or charge. This pilot does not invent root tracking, compatible neutral/anion active spaces, PECs, CBS limits, or uncertainty bounds.
- Spin expectation is checked for the prescribed spin, but this is insufficient to establish the correct physical electronic state.
- No SR+MR energy correction addition is permitted here: MR and SR describe competing electronic reference choices, not an automatically additive correction.
- NEVPT2 here is strongly contracted second-order perturbation and **not** claimed universally benchmark-accurate. Validation at multiple active spaces, basis levels, state manifolds and benchmark diatomics is still necessary.
- All-electron Gaussian orbital basis, C1 for the pilot; spin-free scalar relativistic SFX2C1E is explicitly optional, not SOC. DBOC, SOC, etc remain open.
- No result cache is reused; source HF checkpoint is hashed into the result signature, and new calculations must be explicitly authorized.

## Artemis smoke

`nice -n 19 python scripts/openea_mr_casscf_nevpt2_smoke.py` exercises **neutral LiH** (SA-CASSCF, two CASCI/NEVPT2 roots) and **LiH⁻** (open-shell single root) at STO-3G. These inputs are purely backend test fixtures, **not** science-quality EA or a statement that LiH⁻ is bound. A failed smoke means fix the PySCF integration before any production work.

## Next MR phases (do not automate indiscriminately)

- Explicit active-space candidate generation and validation, orbital occupation analysis, and orbital correspondence through geometry/basis/charge changes.
- Mixed spin/symmetry manifolds where required, spin adaptation and continuous state identity, PEC and asymptotic state completeness.
- Electronic-model/basis convergence with error budget and reference benchmarks; only then permit `MRProductionCapability.VALIDATED_AVAILABLE` for an applicable domain.
- Share the validated spin-free state information with a later state-interaction SOC calculation.

### PySCF 2.14.0 sort_mo boundary (post Artemis smoke finding)

`MRPointRequest.active_orbital_indices` is a tuple for immutable provenance.
The PySCF `CASSCF.sort_mo(caslst, mo_coeff=None, base=1)` implementation
indexes a one-dimensional NumPy mask with `mask[caslst]`. Passing a tuple
therefore performs unintended multidimensional indexing. The execution
adapter must pass `list(active_orbital_indices)`, explicit reconverged
`mf.mo_coeff`, and `base=0`. This addresses the neutral LiH smoke failure
observed on Artemis; it does not certify the downstream CASCI/NEVPT2 stages,
which must still pass a real PySCF smoke run.

### Phase C / Patch 20 addition

Completed MR points additionally retain `inactive_mo_coeff_ao` (the AO
coefficients of doubly occupied *inactive* CASSCF orbitals, including the
zero-column case). Together with the active CASCI 1RDM and optimized active
MO coefficients this reconstructs the total spin-summed 1RDM across distinct
CAS partitions.  See `OPENEA_V1_MR_ACTIVE_SPACE_COMPARISON.md`.  The runner's
scientific validation status does not change.
