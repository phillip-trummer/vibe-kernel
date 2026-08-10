#!/usr/bin/env bash
set -euo pipefail

workspace_path="${1:-.runs/cuco_static_set_sol}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$repo_root/data/cuco_static_set"
cuco_dir="$repo_root/.deps/cuCollections"

uv run python scripts/seed_task.py \
    --workspace "$workspace_path" \
    --data-dir "$data_dir" \
    --task static_set_build_probe_i32 \
    --adapter sol \
    --starting-kernel cuco_static_set_sol \
    --include-dir "$cuco_dir/include" \
    --no-reference-timing \
    --representative-workloads \
        02e63735-36c4-4ef7-9d66-4a003450bf91 \
        11000000-0000-4000-8000-000000000004 \
        11000000-0000-4000-8000-000000000008 \
        11000000-0000-4000-8000-000000000012

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
