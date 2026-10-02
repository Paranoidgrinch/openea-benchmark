# OpenEA v1 — Evidence-driven Method/Basis Advisor

The advisor is not a lookup table `molecule -> lucky functional/basis pair` and it never uses experimental EA values.

It combines chemical-intelligence features with diagnostic evidence and measured basis convergence. PBE, PBE0, TPSSh and B3LYP remain a reconnaissance ensemble for state discovery and functional sensitivity only. They are never majority-voted or averaged into the final EA.

High-accuracy branch policy:
- single-reference bound state -> CCSD(T);
- confirmed multireference character -> CASSCF / SC-NEVPT2 branch;
- near-threshold attachment -> EOM-EA plus stabilization/continuum diagnostics;
- unresolved diagnostics -> fail closed.

Basis decisions remain independent axes: correlation-consistent valence vs core-valence family, relativistically compatible treatment, diffuse level, and cardinal progression.

For the real OH validation series, QZ is contracting but remains above the explicit cardinal target. The advisor must therefore request cardinal number 5 rather than switch DFT functionals.

Only after the relevant high-accuracy branch exists and both cardinal and diffuse convergence are cleared, with no outstanding core-valence or scalar-relativity tests, may the advisor request `PROCEED_COMPONENT_RESOLVED_CBS`.
