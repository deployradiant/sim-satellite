from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import sim_satellite.generation_evaluation as generation_evaluation
from sim_satellite.generation_evaluation import (
    generation_metrics_output_paths,
    render_generation_metrics_markdown,
    resolve_checkpoint_paths,
)


def test_resolve_checkpoint_paths_uses_default_and_deduplicates_globs(tmp_path: Path) -> None:
    default_checkpoint = tmp_path / "default.pt"
    default_checkpoint.write_bytes(b"default")
    first = tmp_path / "model-step-1.pt"
    second = tmp_path / "model-step-2.pt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    assert resolve_checkpoint_paths(
        (),
        (),
        default_checkpoint_path=default_checkpoint,
    ) == (default_checkpoint,)

    assert resolve_checkpoint_paths(
        (first,),
        (str(tmp_path / "model-step-*.pt"),),
        default_checkpoint_path=default_checkpoint,
    ) == (first, second)

    with pytest.raises(ValueError, match="did not match"):
        resolve_checkpoint_paths(
            (),
            (str(tmp_path / "missing-*.pt"),),
            default_checkpoint_path=default_checkpoint,
        )


def test_generation_metrics_output_paths_default_to_artifacts() -> None:
    json_path, markdown_path = generation_metrics_output_paths(None, None)

    assert json_path == Path("artifacts/generation-metrics-comparison.json")
    assert markdown_path == Path("artifacts/generation-metrics-comparison.md")


def test_render_generation_metrics_markdown_formats_comparison_table() -> None:
    table = render_generation_metrics_markdown(
        {
            "results": [
                {
                    "checkpoint_path": "artifacts/model-a.pt",
                    "generation_metrics": {
                        "status": "computed",
                        "sample_count": 60,
                        "mse": 0.123456789,
                        "psnr": 9.5,
                        "ssim": 0.75,
                        "mask_rgb_mean_mae_macro": 0.25,
                        "mask_gradient_mae_macro": None,
                        "mask_boundary_gradient_delta": 0.125,
                    },
                }
            ]
        }
    )

    assert "| checkpoint | status | samples | mse | psnr | ssim |" in table
    assert "| model-a.pt | computed | 60 | 0.123457 | 9.5 | 0.75 | 0.25 |  | 0.125 |" in table


def test_generation_metrics_cli_dry_run_preflights_without_torch_generation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                '{"image": "images/a.png", "mask": "masks/a.png", '
                '"prepared": "prepared/a.pt", "split": "val", "source": "openearthmap"}',
                '{"image": "images/b.png", "mask": "masks/b.png", '
                '"prepared": "prepared/b.pt", "split": "val", "source": "openearthmap"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "checkpoint_path": str(tmp_path / "default.pt"),
                "batch_size": 1,
            }
        ),
        encoding="utf-8",
    )
    checkpoint_a = tmp_path / "checkpoint-a.pt"
    checkpoint_b = tmp_path / "checkpoint-b.pt"
    checkpoint_a.write_bytes(b"placeholder")
    checkpoint_b.write_bytes(b"placeholder")

    result = CliRunner().invoke(
        generation_evaluation.cli,
        [
            str(config_path),
            "--checkpoint",
            str(checkpoint_a),
            "--checkpoint",
            str(checkpoint_b),
            "--samples",
            "2",
            "--steps",
            "60",
            "--validation-split",
            "val",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["checkpoint_paths"] == [str(checkpoint_a), str(checkpoint_b)]
    assert payload["config"]["generation_metrics_samples"] == 2
    assert payload["config"]["generation_metrics_steps"] == 60
    assert payload["config"]["validation_split"] == "val"
    assert sorted(payload["validation_samples"]) == [
        str(tmp_path / "prepared" / "a.pt"),
        str(tmp_path / "prepared" / "b.pt"),
    ]
