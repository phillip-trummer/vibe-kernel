1. **Premature Structural Commitment**: The agent frequently terminates prematurely, rationalizing the choice as hitting a "structural wall," facing "diminishing returns," or "deferring to protect the working kernel." Yet when a human directive forces it to persist or attempt the rewrite, it clears these plateaus and unlocks substantial gains, which confirms the termination is a behavioral policy rather than an objective hardware or algorithmic ceiling.
2. **Self-Poisoning via Free-Form Memory**: Free-form text memory reinforces the agent's confirmation bias. When the agent abandons a path at a temporary bottleneck, it records the unverified conclusion in persistent memory (e.g., that "the remaining gap is structural and bounded by constraints, and further rewrites have low expected-value"). Later sessions reimport these notes as objective truth, cementing the premature convergence and stalling exploration.
The first behavior is real, but I do not think it is the fundamental failure mode.

The broader failure mode is:

> An agent turns a narrowly scoped negative result into a permanent conclusion.

“Strategy X failed” can actually mean several different things:

- This implementation of X was immature.
- X lost against this particular kernel architecture.
- X lost on this workload distribution.
- X lost with this parameterization.
- X was genuinely unsuitable.

The memory should preserve enough objective context for the next agent to distinguish those cases, without storing the previous agent’s conclusion.

## Observed behaviors

### 1. An initially poor implementation can mature into a winner

This is probably the most important behavior.

The tensor-core producer began 1.5–2× slower. The agent did not reject the architecture because profiling exposed specific implementation defects:

1. Severe shared-memory bank conflicts.
2. Shared-load dependency stalls.
3. 39 million integer division/modulo instructions.
4. A staging rewrite finally made the architecture faster.

[Timeline](</pub/scratch/ptrummer/final/vibe-kernel/.runs/mla-sol-simple-09-08/results/timeline.txt:1023>)

This is different from retrying something on a new baseline. The baseline stayed roughly the same; the implementation matured.

Memory implication: one hypothesis may need several attempts before it has received a fair test.

---

### 2. A strategy’s value changes when the surrounding architecture changes

Your first behavior belongs here.

The clearest example is the one-exponential softmax update:

> “I’m revisiting one previously marginal idea under the new architecture…”

It then became clearly beneficial. [Timeline](</pub/scratch/ptrummer/final/vibe-kernel/.runs/mla-sol-simple-09-08/results/timeline.txt:916>)

The agent also explicitly retried eight-warp QK because:

> “the earlier rejection predates these changes.”

[Attached trace](</home/ptrummer/.codex/attachments/2a7b7e1e-598c-4894-853a-fb64a28aff34/pasted-text.txt:28>)

It still lost, but the reasoning was valid.

Memory implication: a result is conditional on the kernel against which it was tested. This is the strongest argument for retaining some automatically captured baseline provenance.

---

### 3. A stable checkpoint can spawn many independent hypotheses

From the same validated kernel, the agent investigated:

- two-block DSM clusters;
- eight-warp QK;
- 16-byte gathers;
- split-policy changes;
- persistent P·V fragments.

These were not one linear optimization path. The agent repeatedly restored the checkpoint and posed another question. [Attached trace](</home/ptrummer/.codex/attachments/2a7b7e1e-598c-4894-853a-fb64a28aff34/pasted-text.txt:24>)

Memory implication: unsuccessful experiments must remain visible even though the current kernel never advanced.

This was the original weakness of “log only accepted improvements.”

---

### 4. One hypothesis often produces multiple implementation attempts

The DSM hypothesis progressed through:

1. Direct remote DSM loading into WMMA.
2. Copying DSM data into local shared memory.
3. Adding a cluster barrier to protect DSM lifetime.
4. Finally reaching a correct performance result.

[Attached trace](</home/ptrummer/.codex/attachments/2a7b7e1e-598c-4894-853a-fb64a28aff34/pasted-text.txt:1>)

These are not three independent architecture experiments. They collectively test one strategy.

Memory implication: we need experiments containing attempts. Logging only the final attempt loses useful implementation history; treating every attempt as a top-level experiment obscures the research question.

---

### 5. Agents conduct parameter sweeps and restore an earlier result

The split-policy sweep followed this pattern:

```text
try chunk 20
restore
try chunk 15
try chunk 14
restore chunk 15
full benchmark
narrow the dispatch region
restore the original validated checkpoint
```

[Attached trace](</home/ptrummer/.codex/attachments/2a7b7e1e-598c-4894-853a-fb64a28aff34/pasted-text.txt:42>)

Memory implication: benchmark chronology and retained working source are different concepts. A checkout or restoration is a meaningful event; “latest benchmark is the state” is not always true.

---

### 6. Results are often workload-dependent rather than simply better or worse

Examples include:

- 16-token splitting improved small/medium but regressed xlarge.
- A cap-aware policy repaired xlarge but initially regressed batch 1.
- A 256-thread merge improved long cases but lost on the full distribution.
- Narrow chunk-15 tuning improved the representative medium workload but worsened the full geomean.

Memory implication: a single geomean or “slower” label is insufficient. Attempts need the full evaluation plus representative outcomes. This is what later enables dispatch or hybrid solutions.

---

### 7. Smoke success is provisional; full evaluation can reverse it

The agent repeatedly observed representative improvements that lost on the full 47-workload suite. The narrow chunk-15 policy is a direct example. [Attached trace](</home/ptrummer/.codex/attachments/2a7b7e1e-598c-4894-853a-fb64a28aff34/pasted-text.txt:60>)

Memory implication: transient smoke checks and durable full experiments should remain distinct. Your proposed rule—`full` requires `change` and records the attempt—is well supported.

---

### 8. Failure has multiple stages

An attempted strategy may:

- fail to compile;
- compile but fault at runtime;
- fail correctness;
- pass correctness but regress;
- improve only some workloads;
- win fully.

The DSM experiment crossed runtime failures before becoming correct-but-slower. The tensor producer crossed slower-but-correct before becoming faster.

Memory implication: “failure” cannot be one terminal status. The objective evaluation status matters, and an experiment should remain active through repair attempts.

---

### 9. Failed experiments can constrain or inspire a later design

The failed cluster and vector-gather experiments showed that the kernel needed:

- preserved occupancy;
- high memory-level parallelism;
- minimal synchronization.

The next session used those constraints to invent a different P·V accumulator design. It did not merely retry one listed idea.

Memory implication: preserving the experiment names, changes, artifacts, and evaluations may be enough. We do not necessarily need agent-authored `finding`; the next agent can synthesize the evidence.

---

### 10. Agents declare exhaustion too early

The timeline contains multiple “all avenues exhausted” or “final checkpoint” declarations, followed by later sessions finding substantial architectural gains.

Memory implication: the memory should record what was tested, not encode an experiment or architecture as permanently exhausted. An experiment ending is not the same as its hypothesis being universally disproven.

---

### 11. Sessions can stop at three different moments

A session may end:

- after restoring the incumbent, with several completed failed experiments;
- midway through implementing an experiment;
- after success, with a proposed next hypothesis but no attempts.

Memory implication: the active experiment and current workspace must make all three handoffs understandable.

## My ranking

The highest-value behaviors are:

1. **An immature strategy may require multiple attempts before evaluation.**
2. **A result is conditional on the surrounding kernel architecture.**
3. **Many independent experiments may be conducted without advancing the incumbent.**
4. **Workload-dependent results can later be combined into a hybrid.**
5. **Sweeps frequently end by restoring an earlier artifact.**

So I would revise your initial statement into two separate behaviors:

```text
- A strategy may look unsuccessful until implementation defects are repaired
  through several attempts.

- A previously rejected strategy may become worthwhile after the surrounding
  kernel architecture changes.
```

The first is more common within a session. The second is crucial across sessions.

The unifying design requirement is not “remember failures.” It is:

> Preserve the scope of every result so that a future agent does not mistake local evidence for a universal conclusion.

This inventory also clarifies the baseline question: baseline provenance is potentially valuable because it scopes a result, not because we need to reconstruct a tree. Whether that provenance deserves an explicit agent-visible field is the next question; we should decide it against these behaviors rather than against graph aesthetics.