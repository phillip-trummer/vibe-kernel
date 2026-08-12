"""Create a structural branch with its first benchmarked experiment."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from kernel_tools.registry import registry

from .log_experiment import _record_experiment


SCHEMA = {
    "name": "create_branch",
    "description": (
        "Explicitly record the fully benchmarked working kernel as a new structural "
        "branch from head. Use this when the experiment establishes a distinct "
        "structural direction worth preserving and revisiting independently from its "
        "base. Use log_experiment when it is another realization of the active "
        "structure, even if many implementation details changed. Benchmark outcome "
        "does not decide this classification. The branch and its first experiment are "
        "created together, including when the result fails or regresses. "
        "When implementing an open hypothesis, pass its hypothesis_id to consume "
        "exactly that hypothesis after the branch is recorded. Omit hypothesis_id "
        "for a spontaneous branch; no hypothesis is then consumed. Use "
        "checkout_experiment first to select another base. Requires "
        "benchmark_kernel(scope='full') on this exact source."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slug": {
                "type": "string",
                "description": (
                    "Short branch and experiment identifier using lowercase letters, "
                    "digits, and underscores. Do not include assigned bN or eN prefixes."
                ),
            },
            "structure": {
                "type": "string",
                "description": (
                    "Concise permanent description of the structural distinction. "
                    "Omit tuning history, measurements, and future plans."
                ),
            },
            "variant": {
                "type": "string",
                "description": (
                    "Concise self-contained description of the first concrete variant "
                    "of this structure. State its parameter, schedule, layout, "
                    "synchronization, or implementation choices; omit measurements "
                    "and conclusions."
                ),
            },
            "hypothesis_id": {
                "type": "string",
                "description": (
                    "Optional open hypothesis id implemented by this branch. Its base "
                    "must equal head. The hypothesis is consumed only after successful "
                    "logging; omit for a spontaneous branch."
                ),
            },
        },
        "required": ["slug", "structure", "variant"],
    },
}


@registry.register(SCHEMA)
def create_branch(
    workspace: Path,
    slug: str,
    structure: str,
    variant: str,
    hypothesis_id: Optional[str] = None,
) -> str:
    return _record_experiment(
        workspace,
        slug=slug,
        variant=variant,
        structure=structure,
        hypothesis_id=hypothesis_id,
    )
