# OpenEA v1 — Real OH/OH- cardinal basis-series smoke

This validation layer extends the successful real OH/OH- electronic-EA smoke
from one aug-cc-pVDZ calculation to a matched cardinal series:

- aug-cc-pVDZ
- aug-cc-pVTZ
- aug-cc-pVQZ

Every basis point is produced by the same previously validated end-to-end
workflow:

    adaptive neutral PEC
    adaptive anion PEC
    equilibrium resolution
    atomic fragment thresholds
    anion binding gate
    electronic EA decision

Only the resulting electronic-EA interval is passed to the generic
`basis_convergence` layer.

The series runner therefore cannot bypass any lower-level gate.

## Validation thresholds

The smoke script exposes the cardinal convergence settings as CLI arguments.
Its defaults are validation-only:

    cardinal increment target = 0.02 eV
    maximum contraction ratio = 0.75

They are not universal production defaults.

The output may be either:

- `CLEARED`, if the DZ->TZ->QZ series is sufficiently small and contracting;
- `NEED_MORE_EVIDENCE / COMPUTE_NEXT_CARDINAL`, if an aug-cc-pV5Z point is
  required;
- `UNRESOLVED`, if the evidence itself is inconsistent.

No basis averaging is performed.

A cleared cardinal series still does not equal a production CBS EA. The next
scientific layer is component-resolved CBS treatment of SCF and correlation
contributions, while diffuse convergence remains a separate axis.
