"""Dependency-free metrics and evidence consistency checks."""

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

LENGTHS = (32, 64, 128)
MODES = ("fp32", "int8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def validate_tasks(tasks: dict[str, Any]) -> None:
    if tasks["schema_version"] != 1 or tasks["calibration"] != []:
        raise ValueError("Expected schema 1 with no calibration for dynamic quantization")
    docs = tasks["documents"]
    queries = tasks["queries"]
    for rows in (docs, queries):
        if not rows or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Nonempty, unique IDs required")
        if any(not r["text"].strip() for r in rows):
            raise ValueError("Empty text")
    ids = {d["id"] for d in docs}
    if any(q["relevant"] not in ids for q in queries):
        raise ValueError("Unknown relevance label")
    if {q["split"] for q in queries} != {"development", "held_out"}:
        raise ValueError("Both disjoint query partitions are required")
    if len({q["text"] for q in queries}) != len(queries):
        raise ValueError("Duplicate query text across partitions")


def timing(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Timings must be finite and positive")
    median = statistics.median(values)
    return {
        "median_ms": median,
        "p95_ms": sorted(values)[math.ceil(len(values) * 0.95) - 1],
        "embeddings_per_second": 1000 / median,
    }


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("Vector dimensions differ or are empty")
    if any(not math.isfinite(x) for x in a + b):
        raise ValueError("Non-finite embedding")
    denominator = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    if denominator == 0:
        raise ValueError("Zero embedding")
    return sum(x * y for x, y in zip(a, b)) / denominator


def quality(tasks: dict[str, Any], vectors: list[list[float]]) -> dict[str, Any]:
    docs, queries = tasks["documents"], tasks["queries"]
    if len(vectors) != len(docs) + len(queries):
        raise ValueError("Embedding count does not match fixtures")
    rows = []
    for query, vector in zip(queries, vectors[len(docs) :]):
        scores = [cosine(vector, doc) for doc in vectors[: len(docs)]]
        order = sorted(range(len(docs)), key=lambda i: (-scores[i], i))
        ranking = [docs[i]["id"] for i in order]
        rows.append(
            {
                "id": query["id"],
                "split": query["split"],
                "scores": scores,
                "ranking": ranking,
                "rank": ranking.index(query["relevant"]) + 1,
            }
        )
    metrics = {}
    for split in ("development", "held_out"):
        ranks = [r["rank"] for r in rows if r["split"] == split]
        metrics[split] = {
            "count": len(ranks),
            "accuracy": sum(r == 1 for r in ranks) / len(ranks),
            "mrr": sum(1 / r for r in ranks) / len(ranks),
        }
    return {"metrics": metrics, "queries": rows}


def summarize(tasks: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    validate_tasks(tasks)
    expected = {(mode, repeat) for mode in MODES for repeat in range(3)}
    if len(runs) != 6 or {(r["mode"], r["repeat"]) for r in runs} != expected:
        raise ValueError("Expected three independent runs per mode")
    summary: dict[str, Any] = {"modes": {}, "drift": {}}
    for mode in MODES:
        selected = [r for r in runs if r["mode"] == mode]
        entries: dict[str, Any] = {}
        for length in LENGTHS:
            cases = [r["lengths"][str(length)] for r in selected]
            if any(len(c["warm_ms"]) != 15 or c["timing_tokens"] != length for c in cases):
                raise ValueError("Timing protocol mismatch")
            for c in cases:
                timing([c["first_ms"]])
                if len(c["vectors"]) != len(tasks["documents"]) + len(tasks["queries"]):
                    raise ValueError("Vector count mismatch")
                if any(len(v) != 384 for v in c["vectors"]):
                    raise ValueError("Expected 384-dimensional embeddings")
                if len(c["token_lengths"]) != len(c["vectors"]):
                    raise ValueError("Token count mismatch")
                if any(n < 2 or n > length for n in c["token_lengths"]):
                    raise ValueError("Context limit violation")
            entries[str(length)] = {
                "warm": timing([x for c in cases for x in c["warm_ms"]]),
                "first_ms": [c["first_ms"] for c in cases],
                "quality": [quality(tasks, c["vectors"])["metrics"] for c in cases],
            }
        summary["modes"][mode] = {
            "lengths": entries,
            "rss_after_load_bytes": [r["rss_after_load_bytes"] for r in selected],
            "peak_rss_bytes": [r["peak_rss_bytes"] for r in selected],
        }
    for length in LENGTHS:
        pairs = []
        for repeat in range(3):
            base, quant = [
                next(r for r in runs if r["mode"] == mode and r["repeat"] == repeat)["lengths"][
                    str(length)
                ]
                for mode in MODES
            ]
            drift = [cosine(a, b) for a, b in zip(base["vectors"], quant["vectors"])]
            bq, qq = [quality(tasks, c["vectors"])["queries"] for c in (base, quant)]
            pairs.append(
                {
                    "repeat": repeat,
                    "minimum_cosine": min(drift),
                    "mean_cosine": statistics.mean(drift),
                    "cosines": drift,
                    "worsened": [b["id"] for a, b in zip(bq, qq) if b["rank"] > a["rank"]],
                    "improved": [b["id"] for a, b in zip(bq, qq) if b["rank"] < a["rank"]],
                    "top1_changed": [
                        b["id"] for a, b in zip(bq, qq) if b["ranking"][0] != a["ranking"][0]
                    ],
                }
            )
        summary["drift"][str(length)] = pairs
    return summary


def verify(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "evidence/manifest.json")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Manifest path escapes repository")
        if digest(path) != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
    tasks = read_json(root / "fixtures/tasks.json")
    runs = [
        read_json(root / f"evidence/{mode}-{repeat}.json") for repeat in range(3) for mode in MODES
    ]
    for run in runs:
        if run["fixture_sha256"] != digest(root / "fixtures/tasks.json"):
            raise ValueError("Run uses different fixtures")
        if run["threads"] != 2 or run["inter_op_threads"] != 1 or run["seed"] != 24:
            raise ValueError("Runtime protocol mismatch")
        if set(run["lengths"]) != {str(n) for n in LENGTHS}:
            raise ValueError("Missing context length")
    actual = summarize(tasks, runs)
    if actual != read_json(root / "evidence/summary.json"):
        raise ValueError("Saved summary does not match raw observations")
    return actual
