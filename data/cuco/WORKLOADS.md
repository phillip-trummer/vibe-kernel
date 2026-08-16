# cuCollections hash-container workloads

The suite contains two kernel task definitions over a shared database-state
catalog. 

## Operations under test

### Static-set membership: `static_set_build_probe_i32`

This definition asks a GPU kernel to perform a database-style membership test:

1. Build a set from one column of 32-bit integer keys.
2. Look up every key from a second column in that set.
3. Produce one Boolean result per lookup: present or absent.

For example, the build column could contain the product IDs in a dimension
table, while the probe column contains product IDs from a much larger sales
table. This primitive is useful for semi-joins, filtering, dictionary checks,
and other database operations that only need to know whether a key exists.

### Static-multiset multiplicity: `static_multiset_build_count_each_i32`

This definition preserves duplicate build keys and asks how many build rows
match each probe:

1. Build a multiset from the same kind of 32-bit key column.
2. Count the occurrences of every probe key in that multiset.
3. Produce one 64-bit integer count per probe.

For example, build keys `[4, 4, 4, 7, 7]` and probes `[4, 7, 9]` produce
counts `[3, 2, 0]`. This operation approximates the per-probe counting needed
to size or plan the output of a duplicate-preserving hash join. It does not
materialize the matching tuple pairs.

## Terms and workload axes

- **Build rows (`build_size`)**: Number of input rows inserted into the
  container.
  Repeated keys count as separate rows.
- **Unique keys (`unique_keys`)**: Number of distinct values among the build
  rows. A low unique-to-build ratio means that the input contains many
  duplicates.
- **Probe rows (`probe_size`)**: Number of membership or multiplicity queries.
- **Hit rate (`hit_rate_percent`)**: Percentage of probe keys that occur in the
  build input. The other probes are misses.
- **Zipf alpha (`zipf_alpha_milli`)**: Controls key-frequency skew. Zero is the
  uniform case. A value of 1250 means Zipf alpha 1.25, where a small number of
  popular keys account for many rows; 1500 is stronger skew.

The generator guarantees that every distinct build key occurs at least once,
fills the remaining build rows with duplicates, and shuffles the rows. Hits
are sampled from the build-key universe. In the current generator, miss keys
are distinct values outside that universe. This makes correctness and the hit
rate exact, but does not yet model repeated or skewed misses from a real
database.

## Shared workload suite

All sizes below are row counts. “Representative” identifies the four cases
used by each definition's fast smoke benchmark and by profiling. Both
definitions have 14 workloads and share 13 axis tuples exactly. Each definition
has its own workload UUIDs so results remain unambiguous.

| # | Informal scenario | Build | Unique | Probe | Hits | Zipf | Purpose |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Tiny distinct set (representative: small) | 4,096 | 4,096 | 4,096 | 50% | 0 | Fast correctness and launch-overhead check. |
| 2 | Tiny duplicate-heavy hot set | 16,384 | 1,024 | 4,096 | 90% | 1.25 | Small, highly reused set with popular keys. |
| 3 | Small distinct balanced case | 65,536 | 65,536 | 65,536 | 50% | 0 | Uniform baseline without build duplicates. |
| 4 | Medium duplicate-heavy case (representative: medium) | 262,144 | 65,536 | 262,144 | 90% | 0 | Separates duplicate pressure from frequency skew. |
| 5 | Probe-heavy selective lookup | 65,536 | 16,384 | 524,288 | 10% | 1.25 | Many lookups, mostly misses, against a skewed build input. |
| 6 | Medium distinct balanced case | 524,288 | 524,288 | 524,288 | 50% | 0 | Larger uniform baseline without duplicates. |
| 7 | Large duplicate-heavy selective case | 1,048,576 | 262,144 | 524,288 | 10% | 0 | Build work dominates more strongly than probe work. |
| 8 | Large hot set (representative: large) | 1,048,576 | 65,536 | 1,048,576 | 90% | 1.25 | High duplication, skew, and hit rate together. |
| 9 | Probe-heavy distinct set | 524,288 | 524,288 | 4,194,304 | 90% | 1.25 | Amortizes set construction over many skewed probes. |
| 10 | Extra-large distinct balanced case | 4,194,304 | 4,194,304 | 4,194,304 | 50% | 0 | Large uniform state with no duplicate build keys. |
| 11 | Extra-large duplicate-heavy case | 4,194,304 | 1,048,576 | 4,194,304 | 90% | 0 | Large state with four build rows per distinct key. |
| 12 | Extra-large hot set (representative: xlarge) | 4,194,304 | 262,144 | 4,194,304 | 50% | set: 1.25; multiset: 0 | The set tests coupled skew; the multiset keeps the same 16× build multiplicity without an impractical match-count cross-product. |
| 13 | Uniform primary-key/foreign-key lookup | 1,048,576 | 1,048,576 | 16,777,216 | 100% | 0 | Production-shaped dimension-to-fact lookup with a 16:1 probe/build ratio. |
| 14 | Skewed primary-key/foreign-key lookup | 1,048,576 | 1,048,576 | 16,777,216 | 100% | 1.5 | The same lookup with a few dimension keys receiving most probes. |

Workloads 13 and 14 approximate a common primary-key/foreign-key database
shape: a unique dimension key column is built once and a much larger fact
column probes it. They differ only in probe-key popularity, making the cost of
skew easier to interpret. They are still synthetic and should not be described
as captured production data.

The one operation-specific replacement is workload 12. With multiset
`count_each`, applying Zipf 1.25 to both a duplicate-heavy build and its probes
makes work scale with the cross-product of their key frequencies; the original
state took several seconds per invocation. The multiset variant therefore uses
uniform frequencies while preserving its sizes, cardinality, hit rate, and 16×
build multiplicity. The original skewed state remains in the static-set suite,
where duplicates do not multiply the output semantics.

## What is timed

For both definitions, reported candidate latency covers GPU container
construction and initialization, insertion of all build rows, and the
definition's probe operation. Input generation, output allocation, and wrapper
latency are outside the timed region. Consequently, the set task answers:

> How quickly can this implementation build a set for this state and test this
> batch of keys?

The multiset task instead answers:

> How quickly can this implementation build a duplicate-preserving container
> for this state and compute every probe's build-side multiplicity?

They do not separately measure a persistent table that is built once and
reused across multiple independent requests. The probe-heavy workloads partly
amortize construction within one invocation, but persistent-state execution
would require a different task contract.

Every Boolean or count output is checked exactly against its definition's
reference implementation. Input generation is also outside profiling
measurements, so NCU profiles the named build-and-probe operation rather than
the synthetic data generator.

## Interpretation boundary

The suite controls important database-state properties—scale, cardinality,
duplicates, hit rate, and skew—but cannot capture every feature of real data.
In particular, it does not currently preserve natural row ordering, temporal
locality, correlated columns, repeated miss keys, or a table reused across
queries. A later production-data suite should store or load captured key
columns while retaining this same correctness and timing contract.
