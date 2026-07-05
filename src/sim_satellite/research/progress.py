"""Aggregate benchmark run records into a shared progress report.

This is the read side of the benchmark harness: it scans the committed run
records, keeps only runs measured on the requested spec's eval hash (the
data/measurement identity — sampler settings deliberately do not fragment the
leaderboard), and reports a leaderboard plus the remaining gap to each
competitiveness target. It needs no GPU or dataset, so any agent can call it
instantly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import click

from sim_satellite.research.spec import (
    DEFAULT_RUNS_DIR,
    RUN_RECORD_VERSION,
    SAMPLER_PROTOCOL_FIELDS,
    BenchmarkSpec,
    BenchmarkTarget,
    benchmark_eval_hash,
    benchmark_protocol_sample_digest,
    load_benchmark_spec_file,
)


def load_run_records(runs_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    ignored: list[str] = []
    if not runs_dir.exists():
        return records, ignored
    for path in sorted(runs_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            ignored.append(path.name)
            continue
        if not isinstance(payload, Mapping) or payload.get("record_version") != RUN_RECORD_VERSION:
            ignored.append(path.name)
            continue
        records.append(dict(payload))
    return records, ignored


def metric_value(record: Mapping[str, Any], metric: str) -> float | None:
    value: Any = record
    for part in metric.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def build_progress_report(
    spec: BenchmarkSpec,
    records: Sequence[Mapping[str, Any]],
    *,
    baseline_run_id: str = "",
) -> dict[str, Any]:
    eval_hash = benchmark_eval_hash(spec.protocol)
    matching = [record for record in records if record.get("eval_hash") == eval_hash]
    _require_matching_sample_digest(spec, matching)
    primary_target = _target_for_metric(spec, spec.primary_metric)
    leaderboard = sorted(
        (_leaderboard_entry(spec, record) for record in matching),
        key=lambda entry: _sort_key(entry["primary_value"], primary_target.direction),
    )
    baseline = _baseline_entry(leaderboard, baseline_run_id)
    best = leaderboard[0] if leaderboard and leaderboard[0]["primary_value"] is not None else None
    return {
        "benchmark": spec.name,
        "primary_metric": spec.primary_metric,
        "eval_hash": eval_hash,
        "runs": len(matching),
        "skipped_other_eval": len(records) - len(matching),
        "leaderboard": leaderboard,
        "baseline_run_id": baseline["run_id"] if baseline else None,
        "best_run_id": best["run_id"] if best else None,
        "primary_delta_vs_baseline": _primary_delta(best, baseline),
        "targets": [_target_report(spec, target, leaderboard) for target in spec.targets],
        "warnings": _report_warnings(spec, matching),
    }


def render_progress_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Benchmark progress: {report['benchmark']}",
        "",
        f"Primary metric: `{report['primary_metric']}` | "
        f"Eval: `{str(report['eval_hash'])[:16]}` | "
        f"Runs: {report['runs']} "
        f"(skipped from other evals: {report['skipped_other_eval']})",
        "",
        "## Targets",
        "",
        "| Metric | Direction | Target | Best | Best run | Met |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for target in report["targets"]:
        best_value = target["best_value"]
        lines.append(
            f"| `{target['metric']}` | {target['direction']} | {target['target']} | "
            f"{_format_value(best_value)} | {target['best_run_id'] or '-'} | "
            f"{'yes' if target['met'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Leaderboard",
            "",
            "| Rank | Run | Experiment | Primary | Sampler | Eval digest | Notes |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for rank, entry in enumerate(report["leaderboard"], start=1):
        digest = str(entry["sample_digest"] or "")[:8]
        lines.append(
            f"| {rank} | {entry['run_id']} | {entry['experiment']} | "
            f"{_format_value(entry['primary_value'])} | {_format_sampler(entry['sampler'])} | "
            f"{digest or '-'} | {entry['notes'] or '-'} |"
        )
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def _leaderboard_entry(spec: BenchmarkSpec, record: Mapping[str, Any]) -> dict[str, Any]:
    data = record.get("data")
    data = data if isinstance(data, Mapping) else {}
    checkpoint = record.get("checkpoint")
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    protocol = record.get("protocol")
    protocol = protocol if isinstance(protocol, Mapping) else {}
    return {
        "run_id": str(record.get("run_id", "")),
        "experiment": str(record.get("experiment", "")),
        "notes": str(record.get("notes", "")),
        "created_at_epoch_utc": record.get("created_at_epoch_utc"),
        "git_commit": str(record.get("git_commit", "")),
        "checkpoint_path": str(checkpoint.get("path", "")),
        "model_parameters": checkpoint.get("model_parameters"),
        "sample_digest": data.get("sample_digest"),
        "manifest_samples_by_source": data.get("manifest_samples_by_source"),
        "sampler": {name: protocol.get(name) for name in SAMPLER_PROTOCOL_FIELDS},
        "primary_value": metric_value(record, spec.primary_metric),
        "metrics": {target.metric: metric_value(record, target.metric) for target in spec.targets},
        "wall_seconds": record.get("wall_seconds"),
    }


def _format_sampler(sampler: Mapping[str, Any]) -> str:
    steps = sampler.get("generation_steps")
    guidance = sampler.get("guidance_scale")
    precision = sampler.get("precision")
    if steps is None and guidance is None:
        return "-"
    return (
        f"{steps or '?'} steps, cfg {guidance if guidance is not None else '?'}, {precision or '?'}"
    )


def _sort_key(value: float | None, direction: str) -> tuple[int, float]:
    if value is None:
        return (1, 0.0)
    return (0, -value if direction == "higher" else value)


def _baseline_entry(
    leaderboard: Sequence[Mapping[str, Any]],
    baseline_run_id: str,
) -> Mapping[str, Any] | None:
    if not leaderboard:
        return None
    if baseline_run_id:
        for entry in leaderboard:
            if entry["run_id"] == baseline_run_id:
                return entry
        raise ValueError(f"Baseline run not found: {baseline_run_id}")
    return min(
        leaderboard,
        key=lambda entry: (
            entry["created_at_epoch_utc"] is None,
            entry["created_at_epoch_utc"] or 0.0,
        ),
    )


def _primary_delta(
    best: Mapping[str, Any] | None,
    baseline: Mapping[str, Any] | None,
) -> float | None:
    if best is None or baseline is None:
        return None
    best_value = best["primary_value"]
    baseline_value = baseline["primary_value"]
    if best_value is None or baseline_value is None:
        return None
    return float(best_value) - float(baseline_value)


def _target_for_metric(spec: BenchmarkSpec, metric: str) -> BenchmarkTarget:
    for target in spec.targets:
        if target.metric == metric:
            return target
    raise ValueError(f"Benchmark spec has no target for metric: {metric}")


def _target_report(
    spec: BenchmarkSpec,
    target: BenchmarkTarget,
    leaderboard: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    best_value: float | None = None
    best_run_id: str | None = None
    for entry in leaderboard:
        value = entry["metrics"].get(target.metric)
        if value is None:
            continue
        if best_value is None or _is_better(value, best_value, target.direction):
            best_value = value
            best_run_id = entry["run_id"]
    met = best_value is not None and not _is_better(target.target, best_value, target.direction)
    if best_value is None:
        remaining = None
    elif target.direction == "higher":
        remaining = max(0.0, target.target - best_value)
    else:
        remaining = max(0.0, best_value - target.target)
    return {
        "metric": target.metric,
        "label": target.label,
        "direction": target.direction,
        "target": target.target,
        "best_value": best_value,
        "best_run_id": best_run_id,
        "met": met,
        "remaining_gap": remaining,
    }


def _is_better(value: float, reference: float, direction: str) -> bool:
    if direction == "higher":
        return value > reference
    return value < reference


def _report_warnings(
    spec: BenchmarkSpec,
    records: Sequence[Mapping[str, Any]],
) -> list[str]:
    warnings: list[str] = []
    digests = {
        str(record.get("data", {}).get("sample_digest", ""))
        for record in records
        if isinstance(record.get("data"), Mapping)
    }
    digests.discard("")
    if len(digests) > 1:
        warnings.append(
            "Runs share a protocol hash but evaluated different sample digests; "
            "the underlying dataset changed between runs and metric deltas are "
            "not a pure model comparison."
        )
    missing_primary = [
        str(record.get("run_id", ""))
        for record in records
        if metric_value(record, spec.primary_metric) is None
    ]
    if missing_primary:
        warnings.append(
            f"Runs missing the primary metric `{spec.primary_metric}`: "
            f"{', '.join(sorted(missing_primary))}"
        )
    return warnings


def _require_matching_sample_digest(
    spec: BenchmarkSpec,
    records: Sequence[Mapping[str, Any]],
) -> None:
    expected = benchmark_protocol_sample_digest(spec.protocol)
    if not expected:
        return
    mismatched = [
        str(record.get("run_id", ""))
        for record in records
        if _record_sample_digest(record) != expected
    ]
    if mismatched:
        raise ValueError(
            "Runs share the eval hash but do not match the spec's pinned sample digest: "
            f"{', '.join(sorted(mismatched))}. Start a new benchmark spec or rerun against "
            "the pinned eval_sample_identities."
        )


def _record_sample_digest(record: Mapping[str, Any]) -> str:
    data = record.get("data")
    if not isinstance(data, Mapping):
        return ""
    value = data.get("sample_digest")
    return value if isinstance(value, str) else ""


def _format_value(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


@click.command(
    "benchmark-progress",
    help="Summarize benchmark run records into a leaderboard and target gaps.",
)
@click.option(
    "--runs-dir",
    default=DEFAULT_RUNS_DIR,
    show_default=True,
    help="Directory containing benchmark run record JSON files.",
)
@click.option("--baseline", "baseline_run_id", default="", help="Baseline run id for deltas.")
@click.option(
    "--markdown",
    "markdown_path",
    default="",
    help="Optional path to also write a markdown leaderboard.",
)
@click.argument(
    "spec_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
def progress_cli(
    runs_dir: str,
    baseline_run_id: str,
    markdown_path: str,
    spec_path: Path,
) -> None:
    try:
        spec = load_benchmark_spec_file(spec_path)
        records, ignored = load_run_records(Path(runs_dir))
        report = build_progress_report(spec, records, baseline_run_id=baseline_run_id)
        report["ignored_files"] = ignored
        if markdown_path:
            output_path = Path(markdown_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(render_progress_markdown(report), encoding="utf-8")
            report["markdown_path"] = str(output_path)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    progress_cli()
