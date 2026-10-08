# OpenEA v1 — Active-space comparison (Phase C / Patch 20)

## Purpose and scope

This is a **method-development diagnostic**, not automatic CAS selection or a
validated multi-reference electron affinity.  The already operational generic
CASSCF → multi-root CASCI → SC-NEVPT2 runner is invoked separately, **with an
explicit cost authorization for each requested CAS model**.  Patch 20 adds the
missing inactive-orbital AO coefficients to every completed result, and a
*read-only* comparison of results from **different explicitly defined active
spaces for the same diatomic electronic system and geometry**.

This is not the full MR uncertainty closure required by G3.  Neither agreement
of two CAS calculations nor a seemingly stable CASCI/NEVPT2 energy proves
that other active orbital choices have been excluded.  Production acceptance,
automatic active-space selection, and SOC state validity remain open.

## Why the inactive orbitals matter

Two active spaces can differ in the division between doubly occupied inactive
orbitals and active orbitals, and even have different CAS electron counts.
Directly comparing the two CASCI **active** 1RDMs is therefore incorrect.
Instead, reconstruct the **total** spin-summed one-particle density in the
same AO basis for every root:

\[
D_{\mu\nu}=2\sum_{i\in \text{inactive doubly occupied}} C_{\mu i} C_{\nu i}
 + \sum_{pq\in\text{active}} C_{\mu p}\,\gamma_{pq}\,C_{\nu q}.
\]

The density is converted to an orthonormal AO representation with the positive
square root of the **actual** AO overlap matrix: `S^1/2 D S^1/2`.  Root
similarities use the normalized Hilbert–Schmidt inner product.  This is a
**candidate fingerprint**, not a many-electron CI overlap or a state-identity
proof.  Fully occupied inert core orbitals can dominate the norm and obscure
small valence differences: use explicit assignment margins and expert review.

## Fail-closed comparison contract

`assess_mr_active_space_pair` requires:

1. Same atoms/charge/spin, geometry, orbital basis mapping, Hamiltonian
   settings, state manifold, and source-HF root.
2. Two genuinely different CAS choices, the same explicitly covered root
   count (at least two for one-body fingerprint discrimination), and full
   completed CASSCF/CASCI/SC-NEVPT2 outputs.
3. Both original source-checkpoint SHA256 values **identical and still valid**;
   both request/settings signatures match the source runner's recorded values.
4. Active and inactive AO orbital blocks present, complete CASCI root 1RDMs,
   physical occupations/electron counts, normalized orbitals and a
   well-conditioned positive AO overlap metric.
5. A uniquely mutual-best root assignment with reviewer-defined similarity
   and separation margins; tied or incomplete assignments are unresolved.

Output can be `CANDIDATE_REVIEW_REQUIRED` with observed per-root CASCI and
SC-NEVPT2 total energy *shifts*, or `UNRESOLVED`.  The shift is **not** a
model-error upper bound, a complete correlation uncertainty, proof of root
identity, proof of CAS convergence, or a release for G3 or scientific EA.

The input thresholds are deliberately specified by the caller, not generated
from molecule-specific hardcodes.  Comparisons of different Hamiltonians,
geometry points, geometrical minima, or basis sets belong to distinct review
and PEC/correction paths, not to this CAS-size sensitivity test.

## Validation

- Synthetic exact one-body densities for CAS(2,2) and CAS(4,3) with distinct
  numbers of inactive doubly occupied orbitals.
- Cross-CAS root swaps, ambiguous root fingerprints, damaged/stale source
  checkpoints, signature/setting differences, AO metric failures and malformed
  density/occupation payloads.
- Artemis **software integration smoke only**: LiH/STO-3G, one neutral geometry,
  CAS(2,2) versus CAS(2,3), two singlet CASCI/NEVPT2 roots.  This is **not** a
  scientific reference for CAS convergence or an EA value.

### Next scientific requirements

A genuine CAS choice needs physical valence/near-degenerate orbital selection,
state and spin-manifold completeness, diagnostics across multiple CAS partitions
and (where appropriate) basis levels, numerical convergence, validated root
identity along neutral and anion PECs, and a defensible error model.  Do not
claim all these issues are solved by two CAS points.
