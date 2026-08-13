# vibe-kernel

Agent tools for GPU kernel optimization using Claude Code or Codex.

> [!NOTE]
> This is research software under active development.

### Default kernel tools

| Tool | Purpose |
| --- | --- |
| `read_source` | Read a kernel source file. |
| `edit_source`, `write_source` | Modify a kernel source file. |
| `benchmark_kernel` | Build, validate, and time the kernel. |
| `benchmark_sweep` | Benchmark several values of one source parameter. |
| `profile_kernel` | Profile the kernel on a representative workload with Nsight Compute. |

Supported benchmark backends:

- [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench)
- [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench)

### Memory tools

Kernel optimization is iterative and the search space is deep enough that no single clever edit wins. The optional memory tools allow the agent to preserve experiments across sessions so later attempts can build on earlier work.

| Tool | Purpose |
| --- | --- |
| `read_memory` | Review previous experiments and ideas. |
| `log_experiment` | Save the current kernel and its benchmark results. |
| `checkout_branch` | Return to a saved alternative. |
| `diff_experiment` | Compare two kernel source code snapshots. |
| `update_idea` | Save or remove an idea for future work. |

Experiments form linear histories. Each branch points to its latest experiment, and forks preserve alternative branch heads. Only branch heads can be checked out.

## Quick start

Requirements: Python 3.12+, `uv`, a CUDA-capable NVIDIA GPU, CUDA Toolkit 13,
and either Claude Code or Codex. Nsight Compute (`ncu`) is required for
profiling.

Install the project environment and check the MCP server from the repository
root:

```bash
uv sync
uv run kernel-tools-mcp --help
```

### 1. Create a workspace

The repository includes example data and setup scripts. Choose one:

```bash
# Start with an empty CUDA scaffold and only the kernel tools
bash mla_sol_stub.sh .runs/my_run
```
```bash
# Use the flashinfer benchmarking backend
# starting from an existing Triton solution
# with experiment memory enabled
# and reference timing disabled
bash mla_flash_triton_memory.sh .runs/my_run
```

These examples deliberately vary the backend, starting kernel, language, workload selection, reference timing, and memory configuration. Mix these options as needed.

The scripts seed and validate the workspace, add shared agent instructions, and
configure Claude Code & Codex.

To use another task from the
[`flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)
dataset, download the full dataset first (about 13 GB):

```bash
uv run python scripts/download_data.py
uv run python scripts/seed_task.py --list
```

### 2. Start an agent

The setup scripts register the MCP server and write shared agent instructions.

```bash
cd .runs/my_run
claude
# codex
```

## Workspace layout

The seeding scripts are convenience. The tools only require the workspace to
include the kernel source files (possibly empty) and task fixtures: kernel
spec, workloads & benchmark configuration.

```text
.runs/my_run/
├── task/
│   ├── definition.json       # flashinfer-trace
│   ├── workloads.jsonl       # flashinfer-trace
│   ├── blob/                 # flashinfer-trace
│   └── benchmark.json        # adapter, hardware, reference timing, build spec and representative workloads
└── src/
    ├── kernel.h
    ├── kernel.cu
    └── main.cpp
```

Validate its structure without building or running the kernel:

```bash
uv run python scripts/validate_workspace.py --workspace .runs/my_run
```

After creating this layout without a setup script, add the MCP server to the
agent client manually. From the repository root, resolve the paths first:

```bash
VIBE_KERNEL_ROOT="$(pwd)"
KERNEL_WORKSPACE="$(realpath .runs/my_run)"
```

For Codex:

```bash
codex mcp add kernel-tools -- \
    uv run --project "$VIBE_KERNEL_ROOT" kernel-tools-mcp \
    --workspace "$KERNEL_WORKSPACE"
```

For Claude Code, run the command inside the workspace so the local MCP entry
is associated with that project:

```bash
cd "$KERNEL_WORKSPACE"
claude mcp add --scope local kernel-tools -- \
    uv run --project "$VIBE_KERNEL_ROOT" kernel-tools-mcp \
    --workspace "$KERNEL_WORKSPACE"
```

Use `codex mcp list` or `claude mcp list` to verify the registration.

Reopen the same directory to continue a previous run.

## Database kernels

Beyond inference kernels, we introduce kernel tasks built on [cuCollections](https://github.com/NVIDIA/cuCollections), NVIDIA's
GPU hash containers. The agent builds a hash set or multiset from a column of
integer keys and answers membership or multiplicity queries against it. Their cost is driven by
key cardinality, duplicate rate, hit rate, and skew, so the workloads sweep those.

The `cuco_*` scripts need the cuCollections headers:

```bash
git clone https://github.com/NVIDIA/cuCollections.git .deps/cuCollections
```
