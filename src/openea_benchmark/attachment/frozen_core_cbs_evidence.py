"""Evidence contract for a corrected frozen-core CBS baseline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

REQUIRED_KEYS = (
    ("neutral", "aug-cc-pvqz"),
    ("anion", "aug-cc-pvqz"),
    ("neutral", "aug-cc-pv5z"),
    ("anion", "aug-cc-pv5z"),
    ("neutral", "d-aug-cc-pv5z"),
    ("anion", "d-aug-cc-pv5z"),
)


@dataclass(frozen=True)
class FrozenCoreEvidenceSummary:
    status: str
    correlation_space: str
    required_points: tuple[str, ...]
    reusable_points: tuple[str, ...]
    missing_points: tuple[str, ...]
    next_actions: tuple[str, ...]
    is_production_ea: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_frozen_core_points(
    points: Iterable[Mapping[str, Any]],
) -> FrozenCoreEvidenceSummary:
    by_key = {
        (str(p["role"]), str(p["basis"])): p
        for p in points
    }
    reusable = []
    missing = []
    for role, basis in REQUIRED_KEYS:
        key = f"{role}:{basis}"
        point = by_key.get((role, basis))
        if point is not None and bool(point.get("reusable")):
            reusable.append(key)
        else:
            missing.append(key)

    if missing:
        status = "PARTIAL"
        actions = (
            "RESUME_ONLY_MISSING_FROZEN_CORE_SINGLE_POINTS",
            "DO_NOT_REBUILD_CBS_YET",
        )
    else:
        status = "READY"
        actions = (
            "REBUILD_VALENCE_FROZEN_CORE_COMPONENT_RESOLVED_CBS",
            "THEN_COMPUTE_SEPARATE_CORE_VALENCE_CORRECTION",
        )

    return FrozenCoreEvidenceSummary(
        status=status,
        correlation_space="FROZEN_CORE_PYSCF_AUTO_CHEMCORE",
        required_points=tuple(f"{r}:{b}" for r, b in REQUIRED_KEYS),
        reusable_points=tuple(reusable),
        missing_points=tuple(missing),
        next_actions=actions,
        is_production_ea=False,
    )
