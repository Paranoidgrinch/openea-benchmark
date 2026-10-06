# OpenEA v1 — Correlation-space correction

## Discovered issue

Historical Stage-3 execution created coupled-cluster objects as:

    mycc = cc.CCSD(cc_mf)

with no `frozen` argument and no call to `set_frozen()`.

PySCF therefore correlated all electrons.  The earlier CBS result was
incorrectly described as a frozen-core valence result.

## Scientific consequence

The numerical all-electron result is retained as diagnostic evidence, but it
must not receive an additional core-valence correction.

For a composite protocol, OpenEA now establishes an explicit valence
frozen-core baseline first.  A core-valence correction will then be computed
separately with a core-valence basis.

## Execution correction

`Stage3ExecutionSettings` gains:

    frozen_core: bool = False

When true, Stage-3 invokes:

    mycc.set_frozen()

which uses PySCF's automatic `elements.chemcore` rule.

The default remains false for backward compatibility with stored historical
evidence.

## OH recovery run

Only six new molecular single points are required at the already validated
fixed reference geometries:

- neutral / anion, aug-cc-pVQZ
- neutral / anion, aug-cc-pV5Z
- neutral / anion, d-aug-cc-pV5Z

No PEC or fragment calculation is repeated.
