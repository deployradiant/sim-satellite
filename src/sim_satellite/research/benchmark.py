"""CLI facade for the flow-matching benchmark harness."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import click

from sim_satellite.research.mask_adherence import (
    load_real_adherence_cache,
    real_adherence_cache_path,
    write_real_adherence_cache,
)
from sim_satellite.research.runner import run_benchmark, write_run_record
from sim_satellite.research.selection import BenchmarkSelection, select_benchmark_samples
from sim_satellite.research.spec import (
    DEFAULT_CACHE_DIR,
    DEFAULT_RUNS_DIR,
    RUN_RECORD_VERSION,
    SAMPLER_PROTOCOL_FIELDS,
    BenchmarkProtocol,
    BenchmarkRunOptions,
    BenchmarkSpec,
    BenchmarkTarget,
    benchmark_eval_hash,
    load_benchmark_spec_file,
    load_benchmark_spec_values,
    sample_identity_digest,
    validate_run_options,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_RUNS_DIR",
    "RUN_RECORD_VERSION",
    "SAMPLER_PROTOCOL_FIELDS",
    "BenchmarkProtocol",
    "BenchmarkRunOptions",
    "BenchmarkSelection",
    "BenchmarkSpec",
    "BenchmarkTarget",
    "benchmark_cli",
    "benchmark_eval_hash",
    "load_benchmark_spec_file",
    "load_benchmark_spec_values",
    "load_real_adherence_cache",
    "real_adherence_cache_path",
    "run_benchmark",
    "sample_identity_digest",
    "select_benchmark_samples",
    "write_real_adherence_cache",
    "write_run_record",
]


def _current_git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _run_with_target(
    spec: BenchmarkSpec,
    options: BenchmarkRunOptions,
    *,
    target: str,
    holdout_only: bool,
    real_adherence_cache: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if target == "local":
        if holdout_only:
            result = select_benchmark_samples(
                spec.protocol,
                manifest_path=options.manifest_path,
            ).payload()
            result["eval_hash"] = benchmark_eval_hash(spec.protocol)
        else:
            result = run_benchmark(
                spec,
                options,
                real_adherence_cache=real_adherence_cache,
                progress=click.echo,
            )
        result["execution_target"] = "local"
        return result

    if target != "modal":
        raise ValueError(f"Unsupported benchmark target: {target}")

    from sim_satellite.modal_app import run_benchmark_target

    return run_benchmark_target(
        spec,
        options,
        holdout_only=holdout_only,
        real_adherence_cache=real_adherence_cache,
    )


@click.command(
    "benchmark",
    help="Score a diffusion checkpoint against a frozen benchmark spec.",
)
@click.option(
    "--target",
    type=click.Choice(["modal", "local"]),
    default="modal",
    show_default=True,
)
@click.option("--checkpoint", "checkpoint_path", default="", help="Diffusion checkpoint to score.")
@click.option("--experiment", default="", help="Short experiment name recorded in the run id.")
@click.option("--notes", default="", help="Free-form notes stored in the run record.")
@click.option("--device", default="cpu", show_default=True, help="Device for --target local.")
@click.option(
    "--runs-dir",
    default=DEFAULT_RUNS_DIR,
    show_default=True,
    help="Directory where the run record JSON is written.",
)
@click.option(
    "--mask-adherence/--no-mask-adherence",
    default=True,
    show_default=True,
    help="Score generated images against conditioning masks with the frozen segmenter.",
)
@click.option(
    "--cache-dir",
    default=DEFAULT_CACHE_DIR,
    show_default=True,
    help="Directory for committed per-protocol caches such as real-image judge scores.",
)
@click.option(
    "--holdout-only",
    is_flag=True,
    help="Print the frozen eval selection and holdout patterns without scoring.",
)
@click.argument(
    "spec_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
def benchmark_cli(
    target: str,
    checkpoint_path: str,
    experiment: str,
    notes: str,
    device: str,
    runs_dir: str,
    mask_adherence: bool,
    cache_dir: str,
    holdout_only: bool,
    spec_path: Path,
) -> None:
    try:
        spec = load_benchmark_spec_file(spec_path)
        options = BenchmarkRunOptions(
            checkpoint_path=checkpoint_path,
            experiment=experiment,
            notes=notes,
            device=device,
            git_commit=_current_git_commit(),
            mask_adherence=mask_adherence,
        )
        real_adherence_cache = None
        if not holdout_only:
            validate_run_options(options)
            real_adherence_cache = load_real_adherence_cache(
                Path(cache_dir),
                benchmark_eval_hash(spec.protocol),
            )
        result = _run_with_target(
            spec,
            options,
            target=target,
            holdout_only=holdout_only,
            real_adherence_cache=real_adherence_cache,
        )
        if not holdout_only:
            cache_payload = result.pop("real_adherence_cache", None)
            if isinstance(cache_payload, Mapping):
                result["real_adherence_cache_path"] = str(
                    write_real_adherence_cache(Path(cache_dir), cache_payload)
                )
            record_path = write_run_record(result, Path(runs_dir))
            result["record_path"] = str(record_path)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    benchmark_cli()
