"""Two ROHF reference branches inside one symmetry block (Patch 38).

Wave-0 AlO X2Sigma+: inside the fixed irrep occupation the ROHF equations have
two solutions (different Al->O charge transfer).  The seed from the DFT scout
converges to one; at some bond lengths an internal instability exists and
following it reaches the other, lower-SCF solution.  A job whose points mix
followed and unfollowed references is therefore a mixture of two curves, and
the continuity review rightly rejects it.

Instead of choosing a branch by hand, both are computed at every grid point
as separate, individually continuous jobs:

* ``<job>__refS`` (SEED_CONTINUOUS): seeded from the scout checkpoint, the
  converged reference is kept even when internally unstable (recorded).
* ``<job>__refF`` (STABILITY_FOLLOWED): starting at the points where following
  happened, propagated from neighbour to neighbour along R (each point seeded
  from the already computed high-level reference next to it), following on.

The parent job is superseded.  Both branches describe the same symmetry-
labelled state; the electronic-EA stage compares them by their CCSD(T)
equilibrium energy and records the spread as reference sensitivity.  Two
branches identical at every point (|dE| below a tolerance) are merged.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adaptive.stage3_execution import (
    SEED_CONTINUOUS_BRANCH, STABILITY_FOLLOWED_BRANCH, Stage3ExecutionRequest,
)
from .adaptive.stage3_naming import checkpoint_artifact_basename

SEED_SUFFIX = "__refS"
FOLLOWED_SUFFIX = "__refF"
SUPERSEDED_STATUS = "SUPERSEDED_BY_REFERENCE_BRANCHES"
MERGED_STATUS = "REFERENCE_BRANCH_IDENTICAL_TO_SEED_BRANCH"


def branch_parent(job_id: str) -> str | None:
    for suffix in (SEED_SUFFIX, FOLLOWED_SUFFIX):
        if job_id.endswith(suffix):
            return job_id[: -len(suffix)]
    return None


def mixed_reference_jobs(requests: Sequence[Stage3ExecutionRequest],
                         records: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Jobs whose completed points mix followed and unfollowed ROHF references."""
    by_job: dict[str, list[int]] = {}
    for req in requests:
        if req.reference_branch is not None:
            continue
        rec = records.get(req.request_id) or {}
        if rec.get("status") != "COMPLETED":
            continue
        by_job.setdefault(req.job_id, []).append(int(rec.get("stability_following_steps") or 0))
    return sorted(job for job, steps in by_job.items()
                  if any(s > 0 for s in steps) and any(s == 0 for s in steps))


def _branch_request(req: Stage3ExecutionRequest, suffix: str, branch: str, **changes) -> Stage3ExecutionRequest:
    return replace(req, request_id=req.request_id + suffix, job_id=req.job_id + suffix,
                   reference_branch=branch, **changes)


def build_branch_requests(parent_requests: Sequence[Stage3ExecutionRequest],
                          records: Mapping[str, Mapping[str, Any]], artifact_dir: Path,
                          ) -> list[Stage3ExecutionRequest]:
    """Seed-branch requests, then followed-branch requests in executable order.

    The followed branch starts at the anchors (points where following
    happened, seeded from their own high-level reference) and walks outwards;
    every other point is seeded from the high-level checkpoint of its
    neighbour towards the nearest anchor, which is computed before it.
    """
    reqs = sorted(parent_requests, key=lambda r: (r.initialization_index, r.r_angstrom))
    seed = [_branch_request(r, SEED_SUFFIX, SEED_CONTINUOUS_BRANCH) for r in reqs]
    followed: list[Stage3ExecutionRequest] = []
    for init in sorted({r.initialization_index for r in reqs}):
        line = [r for r in reqs if r.initialization_index == init
                and (records.get(r.request_id) or {}).get("status") == "COMPLETED"]
        anchors = [i for i, r in enumerate(line)
                   if int(records[r.request_id].get("stability_following_steps") or 0) > 0]
        if not anchors:
            continue
        order = sorted(range(len(line)), key=lambda i: (min(abs(i - a) for a in anchors), i))
        for i in order:
            req = line[i]
            if i in anchors:
                source = records[req.request_id].get("high_level_checkpoint_path") or req.source_checkpoint_path
                link = "REFERENCE_BRANCH_ANCHOR"
            else:
                nearest = min(anchors, key=lambda a: (abs(i - a), a))
                neighbour = line[i + 1] if nearest > i else line[i - 1]
                source = str(Path(artifact_dir) / checkpoint_artifact_basename(neighbour.request_id + FOLLOWED_SUFFIX))
                link = "REFERENCE_BRANCH_NEIGHBOUR_CONTINUATION"
            followed.append(_branch_request(req, FOLLOWED_SUFFIX, STABILITY_FOLLOWED_BRANCH,
                                            source_checkpoint_path=str(source), source_link_status=link))
    return seed + followed


def branches_identical(seed_records: Sequence[Mapping[str, Any]], followed_records: Sequence[Mapping[str, Any]],
                       tolerance_hartree: float = 1.0e-6) -> bool:
    def energies(recs):
        return {round(float(r["r_angstrom"]), 6): r.get("ccsd_t_total_hartree") for r in recs
                if r.get("status") == "COMPLETED"}
    a, b = energies(seed_records), energies(followed_records)
    if not a or set(a) != set(b):
        return False
    return all(a[k] is not None and b[k] is not None and abs(a[k] - b[k]) < tolerance_hartree for k in a)
