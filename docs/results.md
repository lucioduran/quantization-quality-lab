# Measured results

Executed November 9, 2026 on an Intel Xeon W-3223 at 3.50 GHz, macOS x86_64,
48 GiB physical RAM, Python 3.11.13, PyTorch 2.2.2, CPU only. Two intra-op threads
and one inter-op thread. Six fresh workers, three per mode. All seven subprocesses
(one preparation plus six workers) completed successfully. See execution.json for
complete stderr, including library warnings. No paid services were used; energy
and monetary operating costs were not measured.

## Storage and process memory

| Measure | FP32 | Dynamic int8 |
| --- | ---: | ---: |
| Serialized state dictionary bytes | 69,576,640 | 53,223,452 |
| Median RSS after load, MiB | 355.22 | 404.26 |
| Median process peak RSS, MiB | 429.11 | 468.55 |

The int8 file is 23.50% smaller. This is not a 4x compression result: large
embedding tables remain float32. RSS after load is higher for int8. Construction
temporarily allocates float weights, and allocator retention, packed weights,
runtime libraries and process state contribute. These observations do not prove
that int8 intrinsically needs more deployment memory, nor support a memory-saving
claim for this loading path.

## Warm inference

45 raw timed calls per cell, pooled across three workers. Each worker warms up
three times per length. Timing excludes tokenization and loading, includes mean
pooling and normalization, and uses one synthetic repeated-text input. Throughput
is reciprocal median latency, not independently measured concurrent throughput.

| Tokens | FP32 median / p95 ms | Int8 median / p95 ms | Median speedup | FP32 / int8 embeddings/s |
| --- | ---: | ---: | ---: | ---: |
| 32 | 8.495 / 12.692 | 5.536 / 8.277 | 1.53x | 117.7 / 180.6 |
| 64 | 7.674 / 9.367 | 7.343 / 10.498 | 1.05x | 130.3 / 136.2 |
| 128 | 12.549 / 13.716 | 11.188 / 15.176 | 1.12x | 79.7 / 89.4 |

Int8 p95 is worse at 64 and 128 tokens. Length scaling is not monotonic in the
baseline medians, showing runtime/order noise. Do not infer a general performance
curve from these three cases. No confidence interval or universal speed claim.

## First calls

| Tokens | FP32 first calls, ms | Int8 first calls, ms |
| --- | --- | --- |
| 32 | 3137.34, 649.30, 687.65 | 699.01, 699.83, 662.32 |
| 64 | 8.80, 8.76, 7.61 | 7.62, 9.11, 107.60 |
| 128 | 12.90, 13.99, 14.27 | 9.07, 13.72, 13.56 |

Only 32-token observations are process-cold. The first baseline call took 3.14
seconds; this outlier is preserved. Later lengths run after earlier timing and
quality inference. The 107.60 ms int8 shape-first call at 64 tokens is also retained.
Neither disk caches nor the entire machine were isolated.

## Quality and drift

Four development and 24 held-out queries retrieve among 12 documents. Development
accuracy and MRR are 1.0 in both modes at all lengths. All three repeats produce
identical quality results. These are small owned synthetic fixtures, not a standard
benchmark or estimates of population performance.

| Token cap | Held-out accuracy, both | FP32 MRR | Int8 MRR | Minimum embedding cosine | Truncated texts / 40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.7917 | 0.850116 | 0.849124 | 0.953051 | 4 |
| 64 | 0.7917 | 0.849124 | 0.849124 | 0.934400 | 4 |
| 128 | 0.8750 | 0.930556 | 0.930556 | 0.919509 | 0 |

No top-1 result changes under quantization in this fixture set. Rank-level
degradation does occur:

- At 32 tokens, `long4` worsens from rank 6 to 8; `long6` improves from 8 to 7.
- At 64 tokens, `long4` worsens from rank 7 to 8; `long6` improves from 8 to 7.

The long queries share a repeated preamble. At low caps the actual question is
truncated, so their poor retrieval is partly a context failure shared by both
models. At 128 tokens no rank changes occur, although numerical embeddings drift.
No generation, perplexity, GPU behavior, energy use, or production workloads were
evaluated. The corpus and labels were frozen before measurements.

## Evidence map

- `evidence/model.json`: immutable revision, model/source hashes, artifact sizes and hashes.
- `evidence/environment.json`: hardware, packages, timestamps and measured code commit.
- `evidence/execution.json`: ordered child process exits and stderr.
- `evidence/{fp32,int8}-{0,1,2}.json`: full vectors, scores, ranks, token counts, timings and RSS.
- `evidence/summary.json`: recomputable aggregates and individual drift/regression IDs.
- `evidence/manifest.json`: hashes for raw evidence, measured source, protocol and fixtures.

Offline verification proves internal consistency only. It cannot authenticate
the machine or execution. Artifact files are intentionally excluded from Git.
