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
| `profile_kernel` | Profile a representative workload with Nsight Compute; only available for cuda kernels. |

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

Requirements: a CUDA-capable NVIDIA GPU and either Claude Code or Codex.

Install the MCP server from the repository root:

```bash
uv tool install -e .
kernel-tools-mcp --help
```

### Create a workspace

The repository includes example data and two setup scripts. Choose one:

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

The scripts seed and validate the workspace, add shared agent instructions, and
configure Claude Code. Use a new output path for each run.

SOL-ExecBench is not published on PyPI. Before using the SOL example, install a
local checkout into the same `uv` tool environment:

```bash
git clone https://github.com/NVIDIA/SOL-ExecBench.git ../SOL-ExecBench
uv tool install -e . --with-editable ../SOL-ExecBench
```

To use another task from the
[`flashinfer-trace`](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace)
dataset, download the full dataset first (about 13 GB):

```bash
python scripts/download_data.py
python scripts/seed_task.py --list
```

### Start an agent

Claude Code reads the generated project configuration automatically:

```bash
cd .runs/my_run
claude
```

For Codex, register the server from the workspace before starting the agent:

```bash
cd .runs/my_run
codex mcp add kernel-tools -- kernel-tools-mcp --workspace "$PWD"
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
python scripts/validate_workspace.py --workspace .runs/my_run
```

Reopen the same directory to continue a previous run.