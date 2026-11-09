# Design and provenance

The installed package has no runtime dependencies for evidence replay. core.py
contains pure metrics; cli.py verifies SHA-256 hashes and recomputes aggregate
results from raw vectors and timings. benchmark.py is an optional, explicitly
locked measurement harness and imports model libraries only inside measured paths.

Preparation downloads a pinned public model revision into the repository's ignored
cache and creates both state dictionaries. A supervisor retains the shared lock
while preparation and each worker execute serially. Children inherit the lock file
descriptor. The harness checks both inheritance and lock contention before loading
any model library. The shared lock is external infrastructure, not a repository
file, and is neither modified nor replaced here.

Three workers per mode use alternating order: fp32/int8, int8/fp32, fp32/int8.
Workers load only their selected artifact. Mean pooling excludes padding and
normalized embeddings support cosine retrieval. The int8 model uses packed dynamic
Linear kernels; embedding tables remain float32. Serialized file size therefore
cannot be expected to fall by four times.

Hash manifests detect accidental edits. Anyone able to rewrite the evidence and
its manifest can fabricate a self-consistent record. Replay is not attestation.
Source hashes and the measured Git revision identify the code used for execution;
the release includes later tests and documentation. Hardware and timestamps are
self-reported by local operating system APIs. Timing uses perf_counter_ns.

## External references

- [Model card and pooling example](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/tree/4ca70771034acceecb2e72475f72050fcdde4ddc)
- [PyTorch dynamic quantization API](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html)

The recorded PyTorch 2.2.2 runtime is deliberately pinned for the Intel macOS
wheel available on the measured machine. Current documentation may discuss newer
APIs. This project measures the specified legacy eager path, not all quantization
methods or newer PyTorch releases.

The model card supplies mean pooling. This experiment additionally L2-normalizes
the vectors for retrieval; this does not change their cosine similarity.
