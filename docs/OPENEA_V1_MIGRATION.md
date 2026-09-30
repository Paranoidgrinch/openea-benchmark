# OpenEA v1 migration: non-destructive first increment

1. Confirm the active repository commit, remote branch, and cleanliness before
   integrating anything. Do not assume the user's local `5d8cc1c` and public
   branch have identical contents.
2. Keep existing root discovery, determinant fingerprints, manifold, branch
   graph, adaptive DFT PEC, provenance and stage-3 logic intact while the new
   contract is introduced.
3. Add only `src/openea_benchmark/adaptive/`, this documentation, and a new
   `tests/test_adaptive_decision_v1.py`; do not change public APIs or old tests.
4. Run no quantum-chemistry calculations. Test evidence contracts, interval
   arithmetic, sign gating and routing with synthetic records.
5. Next integration: read existing state/PEC records into adapters, *do not*
   duplicate state identity. Design explicit provenance-rich G1/G2/G3 reviewers.
6. Wire real diagnostic providers incrementally. Initially keep G1/G2/G3 OPEN,
   so no synthetic `CLEARED` can leak into production.
7. Test each new provider on existing stored data, then freeze a scientific
   method policy before blind external validation or new prediction work.

## Local audit commands (read-only)

Every terminal block starts in the repository and activates the venv:

```bash
cd /srv/storage/homes/analysis/dschmid/photodetachment/openea-benchmark
source .venv/bin/activate
printf '%s\n' '--- BRANCH/HEAD/STATUS ---'
git remote -v
git status --short --branch
git log -1 --format='%H %ad %s' --date=iso-strict
printf '%s\n' '--- KNOWN MODULES ---'
find src/openea_benchmark -maxdepth 2 -type f -name '*.py' | sort
printf '%s\n' '--- TEST COLLECTION ---'
python -m pytest --collect-only -q
```

After downloading this ZIP and placing it at the repository root (no `exit`):

```bash
cd /srv/storage/homes/analysis/dschmid/photodetachment/openea-benchmark
source .venv/bin/activate
# -n = NEVER overwrite an existing file; review git diff afterward.
unzip -n openea_v1_overlay.zip -d .
python -m pytest -q tests/test_adaptive_decision_v1.py
python -m pytest -q
git status --short
```

No commits, branch changes, installs, pulls, or remote writes are performed by
these commands. The ZIP paths are relative to repository root.

## First actual adapter

`adaptive.adapters.stability_diagnostic_from_root` already consumes the
published `SCFRootRecord` attribute contract without mutating it. In
particular, CANONICALIZED with only `internal_stable=True` and
`external_stable=None` becomes D02 `UNRESOLVED`, not a false blanket stability
approval. Further adapters for manifold/PEC/basis/provenance require inspection
of the corresponding full local modules before wiring them.
