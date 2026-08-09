# vibe-kernel

Agent tools for GPU kernel optimization using Claude Code or Codex.

> [!NOTE]
> This is research software under active development.

### Default kernel tools

| Tool | Purpose |
| --- | --- |
| `read_source` | Read a kernel source file. |
| `edit_source`, `write_source` | Modify a kernel source file. |
| `benchmark_kernel` | Build, validate, and benchmark the kernel. |
| `profile_kernel` | Profile the current implementation on a representative workload with Nsight Compute. |

Supported benchmark backends:

- [flashinfer-bench](https://github.com/flashinfer-ai/flashinfer-bench)
- [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench)

### Memory tools

Kernel optimization is iterative and the search space is deep enough that no single clever edit wins. The optional memory tools preserve that evidence across sessions so later attempts can build on earlier work.

| Tool | Purpose |
| --- | --- |
| `read_memory`, `update_memory` | Read and maintain durable findings, hazards, and hypotheses. |
| `create_branch`, `log_experiment` | Organize and record measured variants. |
| `checkout_experiment`, `diff_experiment` | Restore or compare recorded experiments. |

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

### Create a workspace

The repository includes example data and setup scripts. Choose one:

```bash
# Start with an empty CUDA scaffold and only the kernel tools
bash mla_flash_stub.sh .runs/my_run
```
```bash
# Start from a FlashInfer baseline 
# (the baseline determines the kernel language and build spec) 
# with experiment memory enabled
bash mla_flash_memory.sh .runs/my_run
```

For a SOL workspace, use `mla_sol_stub.sh`.

The minimal cuCollections build-and-probe experiment uses SOL and starts from a
working `cuco::static_set` baseline. Its 14 synthetic workloads vary relation
size, distinct-key count, lookup hit rate, and uniform versus Zipfian key
frequency. Input generation is outside the timed region. Install the
header-only dependency and keep its checkout under `.deps/`:

```bash
git clone https://github.com/NVIDIA/cuCollections.git .deps/cuCollections
```

Then create the workspace:

```bash
bash cuco_static_set_sol.sh .runs/my_cuco_run
```

The scripts seed and validate the workspace, add shared agent instructions, and
configure Claude Code. Use a new output path for each run.

To use another task from the
[`flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)
dataset, download the full dataset first (about 13 GB):

```bash
uv run python scripts/download_data.py
uv run python scripts/seed_task.py --list
```

### Start an agent

Claude Code reads the generated project configuration automatically:

```bash
cd .runs/my_run
claude
```

Codex also reads the generated project configuration automatically:

```bash
cd .runs/my_run
codex
```

## Workspace layout

```text
.runs/my_run/
├── task/
│   ├── definition.json
│   ├── workloads.jsonl
│   ├── blob/
│   └── benchmark.json        # build spec and representative workload selection
└── src/
    ├── kernel.h
    ├── kernel.cu
    └── main.cpp
```

The workspace must include the kernel source files and task fixtures (kernel spec, benchmark configuration & workloads) for the tools to function. 

Validate its structure without building or running the kernel:

```bash
uv run python scripts/validate_workspace.py --workspace .runs/my_run
```

Reopen the same directory to continue a previous run.
