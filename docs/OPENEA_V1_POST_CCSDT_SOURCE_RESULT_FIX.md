# Post-CC source-evidence checkpoint fix

The first OpenEA post-CCSD(T) probe used a filename sanitizer that removed
parentheses from method names.  Consequently both

- `CCSD(T)`
- `CCSDT`

mapped to the token `CCSDT`.

This created a checkpoint-path collision.  It does **not** invalidate the
scientific result because the completed run's aggregate `result.json` stores
the already-derived electron affinities and increments separately:

- `ea_ccsd_t_ev`
- `ea_ccsdt_ev`
- `delta_t3_ev`
- and, at DZ, `ea_ccsdtq_ev` plus `delta_t4_ev`.

The W4 resolver therefore treats the aggregate source `result.json` as the
canonical evidence record for the already completed CCpy probe.

For future post-CC runs, checkpoint tokens are collision-free:

- `CCSD(T)` -> `CCSD_pT`
- `CCSDT`   -> `CCSDT`
- `CCSDTQ`  -> `CCSDTQ`

No expensive source calculation needs to be repeated for the current OH
validation run.
