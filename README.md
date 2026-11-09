# Quantization Quality Lab

[![CI](https://github.com/lucioduran/quantization-quality-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/lucioduran/quantization-quality-lab/actions/workflows/ci.yml)

A reproducible CPU experiment comparing a real pretrained sentence encoder with
its locally quantized int8 version. Smaller weights did not mean lower process
memory on the measured loading path. Raw evidence includes that negative result.

## Problem

Quantization claims often mix model file size, process memory, throughput and
quality. This lab measures each separately and preserves output drift, retrieval
regressions, slow calls and runtime warnings.

## Scope

One 17.4-million-parameter Apache-2.0 model,
[sentence-transformers/paraphrase-MiniLM-L3-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/tree/4ca70771034acceecb2e72475f72050fcdde4ddc),
with PyTorch 2.2.2 dynamic qint8 Linear layers and the fbgemm CPU backend.
The experiment uses 12 synthetic documents, four development queries and 24
held-out queries at 32, 64 and 128 tokens. No calibration is needed for this recipe.

## Quick start

Offline evidence replay uses only Python's standard library. No model download,
inference, account or API key is needed. Use Python 3.11.13 for the recorded environment.

```sh
git clone https://github.com/lucioduran/quantization-quality-lab.git
cd quantization-quality-lab
PYTHONPATH=src python3 -m quantization_quality_lab.cli verify > replay.json
```

The command returns `status: consistent` and `authenticity_verified: false`.
It verifies hashes and recalculates timing, quality and drift from saved observations.

## Usage

Install the small replay package, or use the source command above:

```sh
python3 -m venv .venv-replay
.venv-replay/bin/python -m pip install setuptools==75.8.2 wheel==0.45.1
.venv-replay/bin/python -m pip install --no-build-isolation .
.venv-replay/bin/quantization-quality-lab verify --root .
```

The installed command accepts `verify` and `--root PATH`. Failure returns a nonzero
exit code. [Measured results](docs/results.md) explain the numbers; raw per-query
scores, rankings, embedding vectors and timings are in [evidence](evidence).

## Architecture

- `core.py`: dependency-free validation, cosine retrieval metrics and evidence replay.
- `cli.py`: the offline installed command.
- `benchmark.py`: locked model preparation and serial isolated-process workers.
- `fixtures/tasks.json`: frozen owned texts, relevance labels and split membership.
- `evidence/`: raw observations, package/hardware metadata and SHA-256 manifests.

See [design and provenance](docs/design.md) and the [frozen protocol](docs/protocol.md).
Weights are kept in ignored project-local directories, never in Git.

## Reproduction

Lightweight replay is shown above. It checks internal consistency, not authenticity.
The measured source commit is `93eaf528dd32dfb5b4abf59a2e5c7d1c9b8f67ba`;
fixture creation was committed before execution. The release adds documentation and
regression tests without changing measured source. Exact model/source hashes and
serialized artifact hashes are in [model.json](evidence/model.json). Full installed
versions, hardware, timestamps and wrapper hash are in [environment.json](evidence/environment.json).

For full local measurements, use the recorded Intel macOS environment and the
shared lock wrapper. The harness deliberately refuses unlocked execution and its
hardware collector uses macOS `sysctl`. Other benchmark platforms are untested.
The offline verifier also runs in Linux CI. Dependency installation needs internet;
model preparation downloads about 70 MB of weights from the pinned public revision.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-benchmark.txt setuptools==75.8.2 wheel==0.45.1
.venv/bin/python -m pip install --no-build-isolation -e .
mkdir -p work/reproduction
cp -R src fixtures docs requirements-benchmark.txt work/reproduction/
python3 /Users/luchoduran/Sites/open-source-projects/.coordination/benchmark_lock.py \
  .venv/bin/python -m quantization_quality_lab.benchmark run --root work/reproduction
.venv/bin/quantization-quality-lab verify --root work/reproduction
```

The wrapper is shared infrastructure on the measured host and is not bundled.
Full execution elsewhere requires that infrastructure and recorded path; do not
bypass the lock check. The separate output directory preserves published evidence.
Reusing an output directory with a finished manifest is rejected. Failed subprocess
exits and stderr are written to `execution.json`. Preserve that file before retrying.

New timings and RSS will differ. Serialized hashes may depend on exact library and
serialization versions. To check original artifacts on the measured host:

```sh
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('evidence/model.json').read_text())
for mode, expected in manifest['artifacts'].items():
    path = Path('work/artifacts') / f'{mode}.pt'
    assert path.stat().st_size == expected['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256']
print('Artifact hashes match')
PY
```

## Results

November 9, 2026, Intel Xeon W-3223, 48 GiB RAM, CPU only, two intra-op threads:

| Metric | FP32 | Dynamic int8 |
| --- | ---: | ---: |
| Weight artifact bytes | 69,576,640 | 53,223,452 |
| Median RSS after load | 355.22 MiB | 404.26 MiB |
| Warm median, 32 tokens | 8.495 ms | 5.536 ms |
| Warm median, 64 tokens | 7.674 ms | 7.343 ms |
| Warm median, 128 tokens | 12.549 ms | 11.188 ms |
| Held-out top-1, 128 tokens | 21/24 | 21/24 |

Weights shrink 23.5%, but observed process memory increases. Median inference is
1.05x to 1.53x faster; int8 p95 is worse at 64 and 128 tokens. Top-1 decisions do
not change in this small set. Query `long4` worsens from rank 6 to 8 at 32 tokens
and from 7 to 8 at 64 tokens. Numerical drift remains visible at all lengths.
[Full results](docs/results.md) include raw sample counts, first calls, throughput,
MRR, context truncation and caveats.

## Testing

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/pytest -q
.venv/bin/quantization-quality-lab verify > work/replay.json
.venv/bin/python -m build --no-isolation
```

Tests cover malformed fixtures, tie ordering, numerical errors, tampered hashes,
manifest path traversal, missing/duplicate runs, timing count and context limits,
complete saved rankings, artifact provenance, unlocked benchmark rejection and
observed negative results. CI builds and installs a wheel in a fresh environment
and runs offline replay from outside the checkout. CI never performs inference.
Type checks cover the dependency-free replay layer; lint and tests cover the harness.

## Limitations

One model, machine and small synthetic retrieval set cannot establish general
quality preservation. Only Linear layers are quantized. Timing excludes tokenization
and loading; reciprocal-median throughput is batch-one, not concurrent capacity.
Only the first 32-token call in each worker is process-cold. Length order is fixed,
OS caches are not flushed, and other machine activity is not excluded. All raw
outliers remain. Peak RSS includes construction, allocator retention and runtime
libraries, so it is not pure weight memory. Short caps truncate the substantive
question in four long fixtures. No GPU, generation, perplexity or energy claims.
The legacy pinned PyTorch path is not a recommendation for arbitrary deployments.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Freeze protocols and fixtures before new
measurements, retain failures and use the shared lock for every inference workload.

## License

Code, documentation, evidence and original synthetic fixtures: [MIT](LICENSE).
The separately downloaded model is Apache-2.0, as declared in its pinned model
card. Model weights are not redistributed. See [third-party notices](THIRD_PARTY.md).
