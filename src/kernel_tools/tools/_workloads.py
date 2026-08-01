"""Representative workload conventions shared by benchmark-facing tools.

This module is intentionally flashinfer-free. It only knows our user-facing
representative workload labels. Tools pass in native trace/workload objects and
keep their types intact.
"""
from __future__ import annotations

from typing import Callable, Optional


REPRESENTATIVE_WORKLOAD_LABELS = ("small", "medium", "large", "xlarge")


def representative_items(
    items: list, result_labels: Optional[list[str]] = None
) -> list[tuple[str, object]]:
    """Pair configured representative labels with benchmark results."""
    if result_labels is not None:
        return list(zip(result_labels, items))
    marked = {
        label: item
        for item in items
        if (label := getattr(item, "representative_name", None)) is not None
    }
    missing = [
        label
        for label in REPRESENTATIVE_WORKLOAD_LABELS
        if label not in marked
    ]
    if missing:
        raise ValueError(
            "benchmark results are missing configured representatives: "
            + ", ".join(missing)
        )
    return [
        (label, marked[label])
        for label in REPRESENTATIVE_WORKLOAD_LABELS
    ]


def select_representative_workloads(
    items: list,
    representatives: dict[str, str],
    item_id: Callable[[object], str],
) -> tuple[list, list[str]]:
    """Pick named representative items by configured ID."""
    by_id = {str(item_id(item)): item for item in items}
    missing = [
        f"{label}={workload_id}"
        for label, workload_id in representatives.items()
        if workload_id not in by_id
    ]
    if missing:
        raise ValueError(
            "configured representative workload(s) not found: "
            + ", ".join(missing)
        )
    return (
        [by_id[workload_id] for workload_id in representatives.values()],
        list(representatives),
    )


def representative_item_for_label(
    items: list,
    label: str,
    representatives: dict[str, str],
    item_id: Callable[[object], str],
):
    """Return the item at a representative label's position (types intact)."""
    if label not in REPRESENTATIVE_WORKLOAD_LABELS:
        valid = ", ".join(REPRESENTATIVE_WORKLOAD_LABELS)
        raise ValueError(
            f"invalid representative_workload {label!r}; expected one of: {valid}"
        )
    if not items:
        raise ValueError("no workloads available")
    selected, labels = select_representative_workloads(
        items, representatives, item_id
    )
    return dict(zip(labels, selected))[label]
