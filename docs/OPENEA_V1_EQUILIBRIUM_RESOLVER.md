# OpenEA v1 — Equilibrium Geometry / Energy Resolver

Purpose: turn a converged, identity-cleared Stage-3 *sampled* minimum bracket
into a defensible model/evidence interval for the continuous equilibrium
minimum.

The resolver deliberately does not accept one interpolation as truth.

For one retained sampled minimum it forms a local ensemble of exact
three-point quadratic models. Every model contains the sampled minimum plus a
left and a right high-level PEC point. The nearest-neighbour model is
mandatory, while additional combinations test local model stability.

Resolution requires explicit user/workflow settings for:
- number of side points used,
- minimum number of admissible models,
- minimum positive curvature,
- maximum geometry spread among model vertices,
- maximum energy spread among model vertices,
- numerical point-energy tolerance.

Every admitted model must:
- be convex,
- place its vertex inside the already-converged Stage-3 minimum bracket,
- not place the fitted minimum above the sampled minimum beyond the explicit
  numerical tolerance.

The equilibrium geometry interval remains the Stage-3 bracket. The energy
interval is constructed from the lowest admitted model minimum (minus the
explicit point-energy tolerance) to the sampled high-level minimum (plus that
tolerance). The sampled energy is therefore used only as a conservative upper
bound on the continuous minimum.

The result is labelled `CONVERGENCE_ESTIMATED`. It is not a frequentist
confidence interval and it is not the full final EA uncertainty budget.
Basis-set, correlation, relativistic, SOC, ZPE/nuclear-motion, and other
relevant corrections remain independent downstream evidence.

Fail-closed outcomes include:
- Stage-3 bridge not READY,
- insufficient two-sided PEC data,
- rejected nearest-neighbour model,
- too few admissible models,
- model geometry spread too large,
- model energy spread too large.

No result from this layer assigns a molecular ground state, produces a
production EA, or authorizes pruning.
