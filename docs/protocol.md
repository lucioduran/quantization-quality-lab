# Frozen measurement protocol

Model: sentence-transformers/paraphrase-MiniLM-L3-v2, Apache-2.0.
Use attention-mask mean pooling and L2 normalization, as its model card specifies.
The sentence-transformers configuration limits inputs to 128 tokens even though
the BERT backbone has 512 positions. Enforce the stricter limit, including special tokens.

Recipe: PyTorch 2.2.2 eager dynamic quantization of every torch.nn.Linear to qint8
with the fbgemm CPU backend. Embeddings and layer normalization remain float32.
No calibration or training is required. Calibration partition is empty for this
reason; development and held-out retrieval queries are disjoint. No tuning on
held-out results. Fixed seed 24, CPU intra-op threads 2 and inter-op threads 1.

Freeze fixtures and this protocol in Git before inference. Quality: top-1 retrieval
accuracy and reciprocal rank among the complete fixed document corpus, separately
for development and held-out queries. Ties resolve by corpus order. Record all
scores and rankings, including regressions, and cosine similarity of corresponding
baseline and int8 embeddings. No acceptance threshold or promised speedup.

For each of 32, 64 and 128 token caps, evaluate exactly the same texts in both
modes with truncation, recording actual token lengths and truncation counts.
Also measure a batch-size-one synthetic repeated-text input padded/truncated to
exactly each target token length. This timing input is not a quality fixture.
Measure the first forward call at each shape, then 3 untimed warmup calls
and 15 measured calls. Exclude tokenization and model loading from forward timing.
Report per-call latency, median, p95 (nearest rank), and reciprocal-median
throughput for one embedding, not generation tokens per second.

Create baseline and int8 state dictionaries in a preparation process. Compare
actual file byte sizes with hashes. Each measurement worker constructs its
architecture from config, loads only its selected artifact, and records RSS before
model creation, after load, and after inference, plus OS process peak RSS. Int8
architecture construction temporarily creates random float weights; RSS includes
allocator retention and runtime, so peak RSS is not pure model memory. Three fresh
workers per mode with alternating mode order reduce, but do not remove, order
and cache effects. Only the first 32-token call is process-cold. The 64/128-token first calls follow
other inference and are shape-first observations, not cold starts. None is an OS
disk-cache cold start. Lengths run in ascending order in each worker.
Report memory and timing per process, never claim isolated machine execution.

All model loading, quantization, inference and benchmarks run under the shared
exclusive benchmark_lock.py. Retain raw JSON observations, errors, model file
hashes, exact package versions, hardware, timestamps and code/fixture hashes.
No downloaded weights enter Git. Offline replay checks consistency, not authenticity.
