# OpenEA weekend transfer benchmark

This benchmark is the first deliberate test of whether the workflow
developed using FeH transfers to chemically different diatomics.

FeH is not part of this benchmark because it was the development
stress-test.

The benchmark contains:

- OH: easy positive control
- CN: strongly bound multiple-bond control
- MgH: simpler metal hydride
- AlO: AMS-relevant metal oxide
- N2: unbound negative control

Experimental information is stored for final validation only.

It must not be passed into electronic-state selection, convergence,
method escalation, or root tracking.

The most important outcome is not the smallest mean absolute error.

The benchmark should reveal whether the workflow makes sensible
method-selection decisions:

- ordinary systems should terminate at inexpensive levels;
- problematic correlation should trigger escalation;
- heavy-element corrections should only be activated where useful;
- an unbound anion must be allowed to return UNBOUND.
