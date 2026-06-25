from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sim_satellite.cli import cli
from sim_satellite.sam_finetune import _record_prompt_arrays, _training_records


def test_training_records_can_sample_with_stable_seed(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    records = [{"image": f"image-{index}.png"} for index in range(10)]
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    first = _training_records(records_path, max_records=4, seed=2026)
    second = _training_records(records_path, max_records=4, seed=2026)

    assert first == second
    assert len(first) == 4
    assert {Path(record["image"]).name for record in first} < {
        record["image"] for record in records
    }


def test_record_prompt_arrays_uses_points_and_box() -> None:
    points, labels, box = _record_prompt_arrays(
        {
            "positive_points_xy": [[1, 2], [3, 4]],
            "negative_points_xy": [[5, 6]],
            "bbox_xyxy": [0, 1, 7, 8],
        }
    )

    assert points.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert labels.tolist() == [1, 1, 0]
    assert box.tolist() == [0.0, 1.0, 7.0, 8.0]


def test_record_prompt_arrays_can_match_automatic_mask_generator_prompts() -> None:
    points, labels, box = _record_prompt_arrays(
        {
            "positive_points_xy": [[1, 2], [3, 4]],
            "negative_points_xy": [[5, 6]],
            "bbox_xyxy": [0, 1, 7, 8],
        },
        prompt_mode="positive-point",
    )

    assert points.tolist() == [[1.0, 2.0]]
    assert labels.tolist() == [1]
    assert box is None


def test_cli_exposes_sam_finetune_command() -> None:
    result = CliRunner().invoke(cli, ["sam-finetune", "--help"])

    assert result.exit_code == 0
    assert "Fine-tune SAM2" in result.output
