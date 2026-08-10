#!/usr/bin/env bash
set -euo pipefail

workspace_path="${1:-.runs/cuco_static_multiset_sol}"
data_dir="${VIBE_KERNEL_DATA_DIR:-data/cuco}"
cuco_dir=".deps/cuCollections"

uv run python scripts/seed_task.py \
    --workspace "$workspace_path" \
    --data-dir "$data_dir" \
    --task static_multiset_build_count_each_i32 \
    --adapter sol \
    --starting-kernel cuco_static_multiset_sol \
    --include-dir "$cuco_dir/include" \
    --no-reference-timing \
    --representative-workloads \
        22000000-0000-4000-8000-000000000001 \
        22000000-0000-4000-8000-000000000004 \
        22000000-0000-4000-8000-000000000008 \
        22000000-0000-4000-8000-000000000015

uv run python scripts/validate_workspace.py --workspace "$workspace_path"

uv run python scripts/configure_clients.py \
    --workspace "$workspace_path" \
    --template simple.md \
    --include-task-description \
    --deny-builtins \
    --tools \
        read_source \
        write_source \
        edit_source \
        benchmark_kernel \
        profile_kernel

echo "Workspace ready: $workspace_path"
echo "cuCollections: $cuco_dir"
echo "Claude Code: cd $workspace_path && claude"
echo "Codex: cd $workspace_path && codex"
