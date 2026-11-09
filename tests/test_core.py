import copy
import json
from pathlib import Path

import pytest

from quantization_quality_lab.core import cosine, quality, timing, validate_tasks, verify

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tasks():
    return json.loads((ROOT / "fixtures/tasks.json").read_text())


def test_fixture_partitions(tasks):
    validate_tasks(tasks)
    assert sum(q["split"] == "held_out" for q in tasks["queries"]) == 24
    assert sum(q["split"] == "development" for q in tasks["queries"]) == 4


@pytest.mark.parametrize("mutation", ["duplicate", "empty", "unknown", "split", "calibration"])
def test_reject_invalid_fixtures(tasks, mutation):
    if mutation == "duplicate":
        tasks["queries"].append(copy.deepcopy(tasks["queries"][0]))
    elif mutation == "empty":
        tasks["documents"][0]["text"] = " "
    elif mutation == "unknown":
        tasks["queries"][0]["relevant"] = "missing"
    elif mutation == "split":
        tasks["queries"][0]["split"] = "calibration"
    else:
        tasks["calibration"] = ["not required"]
    with pytest.raises(ValueError):
        validate_tasks(tasks)


def test_metrics_and_stable_tie_order():
    tasks = {
        "documents": [{"id": "a"}, {"id": "b"}],
        "queries": [
            {"id": "dev", "split": "development", "relevant": "a"},
            {"id": "test", "split": "held_out", "relevant": "b"},
        ],
    }
    result = quality(tasks, [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert result["queries"][1]["ranking"] == ["a", "b"]
    assert result["metrics"]["held_out"] == {"count": 1, "accuracy": 0.0, "mrr": 0.5}
    with pytest.raises(ValueError):
        quality(tasks, [[1.0, 0.0]])


@pytest.mark.parametrize("values", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_invalid_times(values):
    with pytest.raises(ValueError):
        timing(values)


def test_latency_statistics():
    assert timing([1.0, 2.0, 3.0, 4.0]) == {
        "median_ms": 2.5,
        "p95_ms": 4.0,
        "embeddings_per_second": 400.0,
    }


@pytest.mark.parametrize("a,b", [([], []), ([1], [1, 2]), ([0], [1]), ([float("nan")], [1])])
def test_invalid_vectors(a, b):
    with pytest.raises(ValueError):
        cosine(a, b)


def test_cosine():
    assert cosine([3, 4], [6, 8]) == 1
    assert cosine([1, 0], [0, 1]) == 0


def test_unlocked_benchmark_is_rejected(monkeypatch, tmp_path):
    from quantization_quality_lab import benchmark

    path = tmp_path / "lock"
    path.touch()
    monkeypatch.setattr(benchmark, "LOCK", path)
    with pytest.raises(RuntimeError, match="wrapper"):
        benchmark.require_lock()


def test_manifest_rejects_escape(tmp_path):
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence/manifest.json").write_text(
        json.dumps({"files": {"../outside": "0" * 64}})
    )
    with pytest.raises(ValueError, match="escapes"):
        verify(tmp_path)


def test_manifest_rejects_changed_bytes(tmp_path):
    (tmp_path / "evidence").mkdir()
    (tmp_path / "item").write_text("modified")
    (tmp_path / "evidence/manifest.json").write_text(json.dumps({"files": {"item": "0" * 64}}))
    with pytest.raises(ValueError, match="SHA-256"):
        verify(tmp_path)
