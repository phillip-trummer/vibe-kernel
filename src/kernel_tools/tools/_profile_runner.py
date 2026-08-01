"""Private child process used by :mod:`kernel_tools.tools.profile_kernel`."""

from __future__ import annotations

import sys
from pathlib import Path

from .profile_kernel import _run_under_ncu


if __name__ == "__main__":
    _run_under_ncu(Path(sys.argv[1]), sys.argv[2])
