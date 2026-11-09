"""Local-only measured execution. Invoke through the shared lock wrapper."""

import argparse
import fcntl
import gc
import importlib.metadata
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

from .core import LENGTHS, MODES, digest, quality, read_json, summarize, validate_tasks, write_json

MODEL_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"
REVISION = "4ca70771034acceecb2e72475f72050fcdde4ddc"
LOCK = Path("/Users/luchoduran/Sites/open-source-projects/.coordination/heavy-benchmark.lock")


def require_lock():
    """Require the wrapper's inherited descriptor and a contended exclusive lock."""
    stat = LOCK.stat()
    inherited = False
    for fd in range(3, 256):
        try:
            candidate = os.fstat(fd)
        except OSError:
            continue
        if (candidate.st_dev, candidate.st_ino) == (stat.st_dev, stat.st_ino):
            inherited = True
            break
    if not inherited:
        raise RuntimeError("Run through the shared benchmark_lock.py wrapper")
    with LOCK.open("r+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(stream, fcntl.LOCK_UN)
        raise RuntimeError("Shared benchmark lock is not held")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def runtime():
    require_lock()
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(24)
    if "fbgemm" not in torch.backends.quantized.supported_engines:
        raise RuntimeError("This recipe requires the fbgemm CPU backend")
    torch.backends.quantized.engine = "fbgemm"
    return torch


def quantize(torch, model):
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def prepare(root):
    torch = runtime()
    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    source = Path(
        snapshot_download(
            MODEL_ID,
            revision=REVISION,
            cache_dir=str(root / ".cache/huggingface"),
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "vocab.txt",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "README.md",
                "sentence_bert_config.json",
                "1_Pooling/config.json",
            ],
        )
    )
    config = read_json(source / "sentence_bert_config.json")
    if config["max_seq_length"] != 128:
        raise ValueError("Unexpected sentence-transformers context limit")
    model = AutoModel.from_pretrained(
        source, local_files_only=True, attn_implementation="eager"
    ).eval()
    artifact_dir = root / "work/artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for mode in MODES:
        converted = model if mode == "fp32" else quantize(torch, model)
        path = artifact_dir / f"{mode}.pt"
        torch.save(converted.state_dict(), path)
        artifacts[mode] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    write_json(root / "work/source.json", {"path": str(source)})
    write_json(
        root / "evidence/model.json",
        {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "license": "Apache-2.0",
            "created_utc": stamp(),
            "artifacts": artifacts,
            "source_files": {
                str(p.relative_to(source)): digest(p)
                for p in sorted(source.rglob("*"))
                if p.is_file()
            },
            "recipe": "torch.ao.quantization.quantize_dynamic, nn.Linear, qint8, fbgemm",
            "parameter_count_fp32": sum(p.numel() for p in model.parameters()),
            "linear_modules": sum(isinstance(m, torch.nn.Linear) for m in model.modules()),
            "context_limit": 128,
        },
    )


def worker(root, mode, repeat):
    started = stamp()
    torch = runtime()
    import psutil
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    process = psutil.Process()
    before = process.memory_info().rss
    source = Path(read_json(root / "work/source.json")["path"])
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    model = AutoModel.from_config(config, attn_implementation="eager").eval()
    if mode == "int8":
        model = quantize(torch, model)
    artifact = root / f"work/artifacts/{mode}.pt"
    expected = read_json(root / "evidence/model.json")["artifacts"][mode]["sha256"]
    if digest(artifact) != expected:
        raise ValueError("Artifact hash mismatch")
    # Load only the locally created, hash-checked state dictionary.
    state = torch.load(artifact, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    del state
    gc.collect()
    after_load = process.memory_info().rss
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tasks = read_json(root / "fixtures/tasks.json")
    validate_tasks(tasks)
    texts = [r["text"] for r in tasks["documents"] + tasks["queries"]]
    full_lengths = [len(tokenizer.encode(t, truncation=False)) for t in texts]
    result = {
        "mode": mode,
        "repeat": repeat,
        "started_utc": started,
        "pid": os.getpid(),
        "seed": 24,
        "threads": torch.get_num_threads(),
        "inter_op_threads": torch.get_num_interop_threads(),
        "artifact_sha256": expected,
        "fixture_sha256": digest(root / "fixtures/tasks.json"),
        "rss_before_model_bytes": before,
        "rss_after_load_bytes": after_load,
        "lengths": {},
    }

    def embed(encoded):
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    with torch.inference_mode():
        for length in LENGTHS:
            encoded = tokenizer(
                "local measurement " * 128,
                return_tensors="pt",
                truncation=True,
                max_length=length,
                padding="max_length",
            )
            start = time.perf_counter_ns()
            embed(encoded)
            first = (time.perf_counter_ns() - start) / 1e6
            for _ in range(3):
                embed(encoded)
            times = []
            for _ in range(15):
                start = time.perf_counter_ns()
                embed(encoded)
                times.append((time.perf_counter_ns() - start) / 1e6)
            vectors, token_lengths = [], []
            for text in texts:
                item = tokenizer(text, return_tensors="pt", truncation=True, max_length=length)
                vectors.append(embed(item)[0].tolist())
                token_lengths.append(int(item["attention_mask"].sum()))
            result["lengths"][str(length)] = {
                "first_ms": first,
                "warm_ms": times,
                "timing_tokens": int(encoded["attention_mask"].sum()),
                "vectors": vectors,
                "token_lengths": token_lengths,
                "untruncated_token_lengths": full_lengths,
                "truncated_texts": sum(n > length for n in full_lengths),
                "quality": quality(tasks, vectors),
            }
    result["rss_after_inference_bytes"] = process.memory_info().rss
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
    result["finished_utc"] = stamp()
    write_json(root / f"evidence/{mode}-{repeat}.json", result)


def run(root):
    require_lock()
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=True)
    if (evidence / "manifest.json").exists():
        raise RuntimeError("Refusing to overwrite published evidence; use a fresh checkout")
    started = stamp()
    env = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_DISABLE_XET="1",
    )
    commands = [("prepare",)]
    for repeat in range(3):
        modes = MODES if repeat % 2 == 0 else tuple(reversed(MODES))
        commands.extend(("worker", "--mode", mode, "--repeat", str(repeat)) for mode in modes)
    executions = []
    for command in commands:
        args = [
            sys.executable,
            "-m",
            "quantization_quality_lab.benchmark",
            *command,
            "--root",
            str(root),
        ]
        began = stamp()
        done = subprocess.run(args, env=env, close_fds=False, text=True, capture_output=True)
        executions.append(
            {
                "args": list(command),
                "started_utc": began,
                "finished_utc": stamp(),
                "returncode": done.returncode,
                "stdout": done.stdout,
                "stderr": done.stderr,
            }
        )
        write_json(evidence / "execution.json", executions)
        if done.returncode:
            raise RuntimeError(f"Execution failed; see {evidence / 'execution.json'}")
        print(f"Completed {' '.join(command)}", flush=True)
    runs = [read_json(evidence / f"{mode}-{repeat}.json") for repeat in range(3) for mode in MODES]
    write_json(evidence / "summary.json", summarize(read_json(root / "fixtures/tasks.json"), runs))
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    write_json(
        evidence / "environment.json",
        {
            "started_utc": started,
            "finished_utc": stamp(),
            "platform": platform.platform(),
            "cpu": cpu,
            "logical_cpus": os.cpu_count(),
            "physical_memory_bytes": memory,
            "python": sys.version,
            "packages": packages,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_status_at_finish": subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ),
            "lock_wrapper": str(LOCK.with_name("benchmark_lock.py")),
            "lock_wrapper_sha256": digest(LOCK.with_name("benchmark_lock.py")),
            "cost": "Local hardware only. No paid services used. Energy cost not measured.",
        },
    )
    files = list(evidence.glob("*.json")) + list((root / "src").rglob("*.py"))
    files += [
        root / "fixtures/tasks.json",
        root / "docs/protocol.md",
        root / "requirements-benchmark.txt",
    ]
    write_json(
        evidence / "manifest.json",
        {
            "schema_version": 1,
            "files": {str(p.relative_to(root)): digest(p) for p in sorted(files)},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "prepare", "worker"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=MODES, default="fp32")
    parser.add_argument("--repeat", type=int, choices=range(3), default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    {
        "run": lambda: run(root),
        "prepare": lambda: prepare(root),
        "worker": lambda: worker(root, args.mode, args.repeat),
    }[args.command]()


if __name__ == "__main__":
    main()
