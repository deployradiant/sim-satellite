from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim_satellite.research.benchmark import (
    BenchmarkProtocol,
    BenchmarkSpec,
    BenchmarkTarget,
    benchmark_eval_hash,
)
from sim_satellite.research.progress import (
    build_progress_report,
    load_run_records,
    metric_value,
    render_progress_markdown,
)

SPEC = BenchmarkSpec(
    name="unit-benchmark",
    protocol=BenchmarkProtocol(samples=4),
    primary_metric="mask_adherence.generated_mean_iou",
    targets=(
        BenchmarkTarget(
            metric="mask_adherence.generated_mean_iou",
            target=0.55,
            direction="higher",
        ),
        BenchmarkTarget(
            metric="generation.mask_rgb_mean_mae_macro",
            target=0.06,
            direction="lower",
        ),
    ),
)


def _record(
    run_id: str,
    *,
    created_at: float,
    mean_iou: float | None,
    rgb_mae: float | None = 0.1,
    eval_hash: str = "",
    sample_digest: str = "digest-a",
) -> dict[str, object]:
    return {
        "record_version": 1,
        "run_id": run_id,
        "experiment": run_id,
        "notes": "",
        "created_at_epoch_utc": created_at,
        "git_commit": "abc",
        "eval_hash": eval_hash or benchmark_eval_hash(SPEC.protocol),
        "checkpoint": {"path": f"artifacts/{run_id}.pt", "model_parameters": 10},
        "data": {"sample_digest": sample_digest, "manifest_samples_by_source": {"oem": 4}},
        "generation": {"mask_rgb_mean_mae_macro": rgb_mae},
        "mask_adherence": (
            {"enabled": True, "generated_mean_iou": mean_iou}
            if mean_iou is not None
            else {"enabled": False}
        ),
        "wall_seconds": 1.0,
    }


def test_metric_value_resolves_dotted_paths() -> None:
    record = _record("a", created_at=1.0, mean_iou=0.4)

    assert metric_value(record, "mask_adherence.generated_mean_iou") == pytest.approx(0.4)
    assert metric_value(record, "generation.mask_rgb_mean_mae_macro") == pytest.approx(0.1)
    assert metric_value(record, "mask_adherence.missing") is None
    assert metric_value(record, "mask_adherence.enabled") is None


def test_progress_report_ranks_runs_and_tracks_targets() -> None:
    records = [
        _record("baseline", created_at=1.0, mean_iou=0.20, rgb_mae=0.10),
        _record("winner", created_at=2.0, mean_iou=0.60, rgb_mae=0.05),
        _record("middle", created_at=3.0, mean_iou=0.40, rgb_mae=0.08),
        _record("other-protocol", created_at=4.0, mean_iou=0.99, eval_hash="different"),
        _record("no-adherence", created_at=5.0, mean_iou=None),
    ]

    report = build_progress_report(SPEC, records)

    assert report["runs"] == 4
    assert report["skipped_other_eval"] == 1
    assert [entry["run_id"] for entry in report["leaderboard"]] == [
        "winner",
        "middle",
        "baseline",
        "no-adherence",
    ]
    assert report["best_run_id"] == "winner"
    assert report["baseline_run_id"] == "baseline"
    assert report["primary_delta_vs_baseline"] == pytest.approx(0.40)

    targets = {target["metric"]: target for target in report["targets"]}
    iou_target = targets["mask_adherence.generated_mean_iou"]
    assert iou_target["met"] is True
    assert iou_target["best_value"] == pytest.approx(0.60)
    assert iou_target["best_run_id"] == "winner"
    assert iou_target["remaining_gap"] == 0.0
    mae_target = targets["generation.mask_rgb_mean_mae_macro"]
    assert mae_target["met"] is True
    assert mae_target["best_value"] == pytest.approx(0.05)

    assert any("missing the primary metric" in warning for warning in report["warnings"])


def test_progress_report_warns_on_diverging_sample_digests() -> None:
    records = [
        _record("a", created_at=1.0, mean_iou=0.2, sample_digest="digest-a"),
        _record("b", created_at=2.0, mean_iou=0.3, sample_digest="digest-b"),
    ]

    report = build_progress_report(SPEC, records)

    assert any("different sample digests" in warning for warning in report["warnings"])


def test_progress_report_respects_explicit_baseline() -> None:
    records = [
        _record("first", created_at=1.0, mean_iou=0.2),
        _record("second", created_at=2.0, mean_iou=0.5),
    ]

    report = build_progress_report(SPEC, records, baseline_run_id="second")
    assert report["baseline_run_id"] == "second"
    assert report["primary_delta_vs_baseline"] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="Baseline run not found"):
        build_progress_report(SPEC, records, baseline_run_id="missing")


def test_progress_report_handles_unmet_lower_targets() -> None:
    records = [_record("only", created_at=1.0, mean_iou=0.30, rgb_mae=0.09)]

    report = build_progress_report(SPEC, records)

    targets = {target["metric"]: target for target in report["targets"]}
    assert targets["mask_adherence.generated_mean_iou"]["met"] is False
    assert targets["mask_adherence.generated_mean_iou"]["remaining_gap"] == pytest.approx(0.25)
    assert targets["generation.mask_rgb_mean_mae_macro"]["met"] is False
    assert targets["generation.mask_rgb_mean_mae_macro"]["remaining_gap"] == pytest.approx(0.03)


def test_render_progress_markdown_lists_targets_and_runs() -> None:
    records = [
        _record("baseline", created_at=1.0, mean_iou=0.20),
        _record("winner", created_at=2.0, mean_iou=0.60, rgb_mae=0.05),
    ]

    markdown = render_progress_markdown(build_progress_report(SPEC, records))

    assert "# Benchmark progress: unit-benchmark" in markdown
    assert "| 1 | winner |" in markdown
    assert "`mask_adherence.generated_mean_iou`" in markdown
    assert "0.6000" in markdown


def test_load_run_records_skips_unknown_files(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    valid = _record("valid", created_at=1.0, mean_iou=0.5)
    (runs_dir / "valid.json").write_text(json.dumps(valid), encoding="utf-8")
    (runs_dir / "broken.json").write_text("{not json", encoding="utf-8")
    (runs_dir / "other.json").write_text(json.dumps({"record_version": 99}), encoding="utf-8")

    records, ignored = load_run_records(runs_dir)

    assert [record["run_id"] for record in records] == ["valid"]
    assert ignored == ["broken.json", "other.json"]

    empty_records, empty_ignored = load_run_records(tmp_path / "missing")
    assert empty_records == []
    assert empty_ignored == []
