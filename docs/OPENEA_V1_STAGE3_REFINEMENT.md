# OpenEA v1 — Adaptive Stage-3 PEC Refinement

This layer acts only after Stage-3 execution, initialization identity, and
geometry continuity have been cleared. It decides which high-level geometries
should be calculated next.

## Decisions

- boundary lowest energy -> extend only that boundary;
- one strict bracketed minimum -> bisect both halves until an explicit bracket
  width target is met;
- multiple strict minima -> retain and refine every bracket;
- non-strict interior minimum / plateau -> resolve its local boundaries;
- insufficient points -> create the minimum additional neighbors needed for
  bracketing evidence.

No fit, equilibrium geometry, EA, ground-state assignment, or state pruning is
performed here.

## Refinement execution seeds

For a new geometry inside the sampled interval, execution requests are created
from both neighboring accepted high-level HF checkpoints. The converged results
must then pass the existing high-level same-geometry identity review again.
For an outward extension, the nearest accepted boundary high-level checkpoint
is used. Geometry continuity must again be demonstrated after execution.

All numerical refinement settings are explicit inputs. The module deliberately
contains no production defaults for extension size, target bracket width,
minimum point spacing, or energy-tie tolerance.
