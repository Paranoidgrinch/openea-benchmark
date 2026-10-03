# OpenEA v1 — Adaptive Cardinal Runner

This closes the gap between cardinal convergence evidence and actual job planning.
The runner evaluates an initial three-point cardinal series, asks the Method/Basis
Advisor for the next action, computes only the requested next cardinal, and
repeats until the cardinal axis is cleared or an explicit maximum is reached.

The runner is generic: molecular calculations are supplied as a callback.
Cardinal convergence remains independent of diffuse, core-valence, relativistic,
and other evidence. `CARDINAL_CLEARED` therefore does not mean production EA.
No finite-basis averaging is performed.
