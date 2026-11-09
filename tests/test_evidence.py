import copy
import json
from pathlib import Path

import pytest

from quantization_quality_lab.core import MODES, digest, quality, summarize, verify

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evidence():
    tasks = json.loads((ROOT / "fixtures/tasks.json").read_text())
    runs = [
        json.loads((ROOT / f"evidence/{mode}-{repeat}.json").read_text())
        for repeat in range(3)
        for mode in MODES
    ]
    return tasks, runs


def test_full_offline_replay():
    assert verify(ROOT)["modes"]["fp32"]["lengths"]["128"]["quality"][0]["held_out"]["count"] == 24


def test_all_saved_rankings_and_artifact_provenance(evidence):
    tasks, runs = evidence
    model = json.loads((ROOT / "evidence/model.json").read_text())
    for run in runs:
        assert run["artifact_sha256"] == model["artifacts"][run["mode"]]["sha256"]
        assert run["fixture_sha256"] == digest(ROOT / "fixtures/tasks.json")
        assert run["peak_rss_bytes"] >= run["rss_after_load_bytes"] > 0
        for length, case in run["lengths"].items():
            assert quality(tasks, case["vectors"]) == case["quality"]
            assert case["token_lengths"] == [
                min(n, int(length)) for n in case["untruncated_token_lengths"]
            ]
            assert case["truncated_texts"] == sum(
                n > int(length) for n in case["untruncated_token_lengths"]
            )


@pytest.mark.parametrize(
    "change", ["missing_run", "duplicate_run", "samples", "context", "dimension"]
)
def test_protocol_regressions_are_rejected(evidence, change):
    tasks, original = evidence
    runs = copy.deepcopy(original)
    if change == "missing_run":
        runs.pop()
    elif change == "duplicate_run":
        runs[-1] = runs[0]
    elif change == "samples":
        runs[0]["lengths"]["32"]["warm_ms"].pop()
    elif change == "context":
        runs[0]["lengths"]["32"]["token_lengths"][0] = 129
    else:
        runs[0]["lengths"]["32"]["vectors"][0].pop()
    with pytest.raises(ValueError):
        summarize(tasks, runs)


def test_observed_negative_outcomes_remain_visible(evidence):
    tasks, runs = evidence
    report = summarize(tasks, runs)
    assert report["drift"]["32"][0]["worsened"] == ["long4"]
    assert report["drift"]["64"][0]["worsened"] == ["long4"]
    for length in ("32", "64", "128"):
        assert report["drift"][length][0]["top1_changed"] == []
    assert min(report["modes"]["int8"]["rss_after_load_bytes"]) > max(
        report["modes"]["fp32"]["rss_after_load_bytes"]
    )
