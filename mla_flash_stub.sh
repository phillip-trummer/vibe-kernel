#!/usr/bin/env bash
set -euo pipefail

workspace_path="${1:-.runs/my_run}"
data_dir="${VIBE_KERNEL_DATA_DIR:-data/mla_paged}"

# Create task/, benchmark.json, and a TVM-FFI CUDA scaffold under src/.
uv run python scripts/seed_task.py \
    --workspace "$workspace_path" \
    --data-dir "$data_dir" \
    --task mla_paged_decode_h16_ckv512_kpe64_ps1 \
    --adapter flashinfer \
    --no-reference-timing \
    --stub cuda \
    --cuda-binding tvm-ffi \
    --auto-representative-workloads

uv run python scripts/validate_workspace.py --workspace "$workspace_path"

# Configure clients without experiment memory or branching tools.
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
        benchmark_sweep \
        profile_kernel

echo "Workspace ready: $workspace_path"
echo "Claude Code: cd $workspace_path && claude"
echo "Codex: cd $workspace_path && codex"
