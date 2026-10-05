# OpenEA v1 — CBS component checkpoint audit

Before any new CBS calculation, OpenEA must determine which component-resolved
electronic energies already exist.

The audit reads trusted local OpenEA stage checkpoints only and extracts, for
every completed Stage-3 point:

- HF/SCF energy;
- CCSD correlation energy;
- CCSD total energy;
- perturbative triples `(T)` correction;
- total CCSD(T) energy;
- geometry and request provenance.

For CBS planning, `aug-cc-pVQZ` and `aug-cc-pV5Z` remain the required cardinal
evidence in v1.  `d-aug-cc-pV5Z` is tracked separately as the high-cardinal
diffuse reference and is never silently substituted into the cardinal
extrapolation.

The audit performs no electronic-structure calculation and no CBS
extrapolation.  Its purpose is to prevent expensive duplicate work.

Workflow position:

    cardinal convergence       CLEARED
    diffuse convergence        CLEARED
             |
             v
    component checkpoint audit <-- this patch
             |
       missing evidence?
        /            \
      yes             no
      |               |
 targeted points      component-resolved CBS
