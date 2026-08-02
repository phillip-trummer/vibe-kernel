#!/usr/bin/env bash
set -euo pipefail

workspace_path="${1:-.runs/my_run}"
data_dir="${VIBE_KERNEL_DATA_DIR:-data/mla_paged}"

# Create task/, benchmark.json, and seed src/ with the selected baseline solution.
uv run python scripts/seed_task.py \
    --workspace "$workspace_path" \
    --data-dir "$data_dir" \
    --task mla_paged_decode_h16_ckv512_kpe64_ps1 \
    --adapter flashinfer \
    --baseline flashinfer_wrapper_03f7b0 \
    --representative-workloads \
        990b57e3-2975-41a1-be67-ecd1ba020887 \
        787d2d2f-548c-46ab-9ded-55fd30b1de20 \
        fd4b2558-ee4f-4d9e-ab3f-7a8333db6340 \
        5bef8d88-0f74-4ccb-a256-b02842951df3

uv run python scripts/validate_workspace.py --workspace "$workspace_path"

# Benchmark and pin the target before the run starts.
uv run python scripts/seed_memory.py \
    --workspace "$workspace_path" \
    --target "$data_dir/solutions/vibe-kernel/opus4.8-25-07_sol.json"

# Add shared instructions, Claude Code permissions, and project MCP config.
uv run python scripts/configure_clients.py \
    --workspace "$workspace_path" \
    --template memory.md \
    --deny-builtins

echo "Workspace ready: $workspace_path"
echo "Claude Code: cd $workspace_path && claude"
echo "Codex: cd $workspace_path && codex"
