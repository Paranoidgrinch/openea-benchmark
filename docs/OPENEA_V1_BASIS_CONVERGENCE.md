# OpenEA v1 — Electronic basis / diffuse convergence evidence

This layer implements D07-style convergence evidence without basis averaging.

Two independent questions are audited.

## 1. Cardinal convergence

At a fixed augmentation level, at least three consecutive cardinal numbers are
required.  For the highest three points the layer evaluates:

- the interval-aware latest change,
- the direction of the last two central changes,
- the contraction ratio of those central changes.

Convergence is cleared only when the latest change is below an explicit target
and the sequence is contracting within an explicit ratio threshold.

No production thresholds are hard-coded.  `BasisConvergenceSettings` has no
scientific defaults.

A cleared residual estimate is deliberately conservative:

    residual <= latest interval-aware change / (1 - contraction ratio)

and expands the highest-cardinal EA interval.  It is labelled
`CONVERGENCE_ESTIMATED`, not a statistical confidence interval.

## 2. Diffuse convergence

At one cardinal number the minimum evidence is matched non-augmented and
augmented results.

If the non-aug -> aug change is already below the explicit diffuse target and
policy does not force another diffuse shell, diffuse convergence may clear.

If the shift is large, or if policy explicitly requires it, a double-augmented
point is required.  The aug -> d-aug increment must then be both small and
contracting.

This allows the future Method Advisor to force d-aug evidence for
diffuse/near-threshold attachment character without forcing it for every
well-bound anion.

## Scientific boundary

This module does **not** yet implement a production CBS extrapolation.

In particular it does not:
- average DZ/TZ/QZ results,
- extrapolate the total CCSD(T) EA with an arbitrary single formula,
- set uncomputed corrections to zero,
- conflate cardinal convergence with diffuse convergence.

The next high-accuracy step is component-resolved CBS evidence, where SCF and
correlation contributions can be treated independently before the resulting
neutral/anion energies are propagated to EA.
