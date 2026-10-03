# OpenEA v1 — Adaptive Diffuse Runner

After the real OH/OH- cardinal axis cleared at X=5, the remaining basis action
was `START_DIFFUSE_SERIES`.

The diffuse runner treats this axis independently. At fixed cardinal X it
starts from matched non-augmented and singly augmented calculations:

    cc-pVXZ
    aug-cc-pVXZ

The generic diffuse-convergence policy decides whether aug is sufficient. If
not, the next requested basis is:

    d-aug-cc-pVXZ

The runner never synthesizes diffuse exponents. Backend basis resolution is
explicit. Missing d-aug support is an execution blocker, not a reason to
silently substitute another basis.

`DIFFUSE_CLEARED` does not produce a production EA. Cardinal convergence,
diffuse convergence, CBS treatment, ZPE and downstream corrections remain
separate evidence layers.
