from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image

from sim_satellite import web
from sim_satellite.data.loader import SatelliteSample
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES, OPENEARTHMAP_RGB
from sim_satellite.data.prepared import write_prepared_sample
from sim_satellite.infer import GenerationConfig
from sim_satellite.model import ModelConfig


def test_web_state_and_preview_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _ = _write_web_fixture(tmp_path)
    app = web.create_app(config)

    state = _request(app, "GET", "/api/state")
    assert state.status_code == 200
    payload = state.json()
    assert "checkpointPath" not in payload
    assert "checkpointGlob" not in payload
    assert "checkpointAvailable" not in payload
    assert payload["checkpoints"] == [
        {
            "id": "primary",
            "label": config.checkpoint_label,
            "path": config.checkpoint_path,
            "glob": "",
            "available": True,
        }
    ]
    assert payload["samples"][0]["name"] == "tile"
    assert payload["classes"][0]["name"] == "unlabeled"
    assert payload["evaluation"]["votes"] == 0

    mask = _request(app, "GET", "/api/mask", query_string="index=0")
    assert mask.status_code == 200
    assert mask.headers["content-type"] == "image/png"
    assert mask.content.startswith(b"\x89PNG")

    image = _request(app, "GET", "/api/image", query_string="index=0")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    with Image.open(BytesIO(image.content)) as preview:
        assert preview.size == (config.display_size, config.display_size)
        assert preview.mode == "RGB"

    asset = _request(app, "GET", "/static/evaluation.js")
    assert asset.status_code == 200
    assert b"createEvaluationController" in asset.content


def test_web_sam_playground_state_and_overlays(tmp_path: Path) -> None:
    config, mask_path = _write_web_fixture(tmp_path)
    Image.new("RGB", (2, 2), OPENEARTHMAP_RGB["building"]).save(mask_path)
    data_dir = Path(config.manifest_path).parent
    labeled_image_path = data_dir / "images" / "tile.png"
    unlabeled_image_path = data_dir / "unlabeled" / "loose.png"
    unlabeled_manifest_path = data_dir / "unlabeled-images.jsonl"
    unlabeled_image_path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), (50, 90, 130)).save(unlabeled_image_path)
    unlabeled_manifest_path.write_text(
        json.dumps(
            {
                "image": "unlabeled/loose.png",
                "split": "unlabeled",
                "source": "openearthmap-unlabeled",
                "region_key": "loose-region",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    labeled_baseline = tmp_path / "sam" / "labeled-baseline" / "proposals.jsonl"
    labeled_trained = tmp_path / "sam" / "labeled-trained" / "proposals.jsonl"
    unlabeled_baseline = tmp_path / "sam" / "unlabeled-baseline" / "proposals.jsonl"
    unlabeled_trained = tmp_path / "sam" / "unlabeled-trained" / "proposals.jsonl"
    _write_sam_proposals(labeled_baseline, labeled_image_path, [255, 0, 0, 0])
    _write_sam_proposals(labeled_trained, labeled_image_path, [255, 255, 0, 0])
    _write_sam_proposals(unlabeled_baseline, unlabeled_image_path, [0, 255, 0, 0])
    _write_sam_proposals(unlabeled_trained, unlabeled_image_path, [0, 255, 255, 0])
    comparison_path = tmp_path / "sam" / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "baseline": {"mean_best_iou": 0.25},
                "trained": {"mean_best_iou": 0.5},
                "delta": {
                    "mean_best_iou": 0.25,
                    "recall_by_threshold": {"0.50": 0.2},
                },
            }
        ),
        encoding="utf-8",
    )
    config = web.WebConfig(
        checkpoint_path=config.checkpoint_path,
        manifest_path=config.manifest_path,
        display_size=4,
        sam_unlabeled_manifest_path=str(unlabeled_manifest_path),
        sam_comparison_path=str(comparison_path),
        sam_labeled_baseline_proposals_path=str(labeled_baseline),
        sam_labeled_trained_proposals_path=str(labeled_trained),
        sam_unlabeled_baseline_proposals_path=str(unlabeled_baseline),
        sam_unlabeled_trained_proposals_path=str(unlabeled_trained),
    )
    app = web.create_app(config)

    page = _request(app, "GET", "/sam")
    assert page.status_code == 200
    assert b"SAM Playground" in page.content

    state = _request(app, "GET", "/api/sam/state")
    assert state.status_code == 200
    payload = state.json()
    assert payload["datasets"]["labeled"]["samples"][0]["name"] == "tile"
    assert payload["datasets"]["unlabeled"]["samples"][0]["name"] == "loose"
    assert payload["proposalSources"]["labeled"]["baseline"]["available"] is True
    assert payload["proposalSources"]["unlabeled"]["trained"]["records"] == 1
    assert payload["comparison"]["value"]["delta"]["mean_best_iou"] == 0.25
    assert "sam-compare-models" in payload["commands"]["labeledComparison"]

    image = _request(
        app,
        "GET",
        "/api/sam/image",
        query_string="dataset=unlabeled&index=0",
    )
    assert image.status_code == 200
    with Image.open(BytesIO(image.content)) as preview:
        assert preview.size == (4, 4)

    label_overlay = _request(
        app,
        "GET",
        "/api/sam/overlay",
        query_string="dataset=labeled&source=label&index=0",
    )
    assert label_overlay.status_code == 200
    assert _has_nontransparent_pixels(label_overlay.content)

    proposal_overlay = _request(
        app,
        "GET",
        "/api/sam/overlay",
        query_string="dataset=unlabeled&source=trained&index=0",
    )
    assert proposal_overlay.status_code == 200
    assert _has_nontransparent_pixels(proposal_overlay.content)

    invalid_label = _request(
        app,
        "GET",
        "/api/sam/overlay",
        query_string="dataset=unlabeled&source=label&index=0",
    )
    assert invalid_label.status_code == 400


def test_web_create_app_does_not_load_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _write_web_fixture(tmp_path)

    def broken_load_generator(checkpoint_path: str, *, device: str) -> object:
        raise RuntimeError("stale checkpoint")

    monkeypatch.setattr(web, "load_generator", broken_load_generator)

    app = web.create_app(config)
    state = _request(app, "GET", "/api/state")

    assert state.status_code == 200


def test_web_generate_uses_generation_only_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, mask_path = _write_web_fixture(tmp_path)
    received: dict[str, GenerationConfig] = {}
    loaded_generator = object()
    load_calls: list[tuple[str, str]] = []

    def fake_load_generator(checkpoint_path: str, *, device: str) -> object:
        load_calls.append((checkpoint_path, device))
        return loaded_generator

    def fake_generate(generator: object, generation_config: GenerationConfig) -> SimpleNamespace:
        assert generator is loaded_generator
        received["config"] = generation_config
        return SimpleNamespace(image=object())

    monkeypatch.setattr(web, "load_generator", fake_load_generator)
    monkeypatch.setattr(web, "generate_image_with_model", fake_generate)
    monkeypatch.setattr(web, "_image_tensor_png", lambda image, display_size: b"png")
    app = web.create_app(config)
    assert load_calls == []

    response = _request(
        app,
        "POST",
        "/api/generate",
        json_body={"index": 0, "seed": 42, "steps": 2},
    )

    assert response.status_code == 200
    assert response.json()["image"] == "data:image/png;base64,cG5n"
    generation_config = received["config"]
    assert isinstance(generation_config, GenerationConfig)
    assert set(generation_config.__dataclass_fields__) == {
        "seed",
        "mask_path",
        "steps",
        "guidance_scale",
    }
    assert generation_config.seed == 42
    assert generation_config.steps == 2
    assert generation_config.mask_path == str(mask_path)
    assert load_calls == [(Path(config.checkpoint_path), config.device)]


def test_web_generate_uses_requested_comparison_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _write_web_fixture(tmp_path)
    comparison_checkpoint = tmp_path / "main-checkpoint.pt"
    comparison_checkpoint.write_bytes(b"comparison")
    config = web.WebConfig(
        checkpoint_path=config.checkpoint_path,
        comparison_checkpoint_path=str(comparison_checkpoint),
        manifest_path=config.manifest_path,
        display_size=4,
    )
    load_calls: list[Path] = []

    def fake_load_generator(checkpoint_path: Path, *, device: str) -> SimpleNamespace:
        load_calls.append(checkpoint_path)
        return SimpleNamespace(checkpoint_path=checkpoint_path)

    def fake_generate(generator: SimpleNamespace, _: GenerationConfig) -> SimpleNamespace:
        return SimpleNamespace(image=object(), checkpoint_path=generator.checkpoint_path)

    monkeypatch.setattr(web, "load_generator", fake_load_generator)
    monkeypatch.setattr(web, "generate_image_with_model", fake_generate)
    monkeypatch.setattr(web, "_image_tensor_png", lambda image, display_size: b"png")
    app = web.create_app(config)

    response = _request(
        app,
        "POST",
        "/api/generate",
        json_body={"index": 0, "checkpoint": "comparison"},
    )

    assert response.status_code == 200
    assert response.json()["checkpointId"] == "comparison"
    assert response.json()["checkpointLabel"] == config.comparison_checkpoint_label
    assert load_calls == [comparison_checkpoint]


def test_web_records_evaluation_vote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _write_web_fixture(tmp_path)
    log_path = tmp_path / "votes.jsonl"
    config = web.WebConfig(
        checkpoint_path=config.checkpoint_path,
        manifest_path=config.manifest_path,
        display_size=4,
        evaluation_log_path=str(log_path),
    )
    monkeypatch.setattr(web.time, "time", lambda: 1234567890.9)
    app = web.create_app(config)

    response = _request(
        app,
        "POST",
        "/api/evaluation/votes",
        json_body={
            "task": "real_fake",
            "trialId": "trial-1",
            "sessionId": "session-1",
            "sampleIndex": 0,
            "sampleName": "tile",
            "seed": 42,
            "choice": "fake",
            "confidence": 4,
            "shown": {
                "kind": "generated",
                "checkpointId": "primary",
                "checkpointLabel": config.checkpoint_label,
                "seed": 42,
            },
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "createdAtEpochUtc": 1234567890,
        "votes": 1,
    }
    [line] = log_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    assert record["createdAtEpochUtc"] == 1234567890
    assert record["task"] == "real_fake"
    assert record["trialId"] == "trial-1"
    assert record["correctChoice"] == "fake"
    assert record["shown"]["checkpointId"] == "primary"


def test_web_rejects_invalid_evaluation_vote_shape(tmp_path: Path) -> None:
    config, _ = _write_web_fixture(tmp_path)
    app = web.create_app(config)

    response = _request(
        app,
        "POST",
        "/api/evaluation/votes",
        json_body={
            "task": "real_fake",
            "trialId": "trial-1",
            "sessionId": "session-1",
            "sampleIndex": 0,
            "choice": "left",
            "confidence": 4,
            "left": {"kind": "generated", "checkpointId": "primary"},
            "right": {"kind": "real"},
        },
    )

    assert response.status_code == 422


def test_web_state_uses_latest_checkpoint_glob(tmp_path: Path) -> None:
    config, _ = _write_web_fixture(tmp_path)
    Path(config.checkpoint_path).unlink()
    first_checkpoint = tmp_path / "model-step-1.pt"
    latest_checkpoint = tmp_path / "model-step-10.pt"
    first_checkpoint.write_bytes(b"first")
    latest_checkpoint.write_bytes(b"latest")
    app = web.create_app(
        web.WebConfig(
            checkpoint_path=str(tmp_path / "missing.pt"),
            checkpoint_glob=str(tmp_path / "model-step-*.pt"),
            manifest_path=config.manifest_path,
            display_size=4,
        )
    )

    state = _request(app, "GET", "/api/state")

    assert state.status_code == 200
    checkpoint = _checkpoint_payload(state.json(), "primary")
    assert checkpoint["path"] == str(latest_checkpoint)
    assert checkpoint["available"] is True


def test_web_checkpoint_glob_waits_for_first_checkpoint(tmp_path: Path) -> None:
    config, _ = _write_web_fixture(tmp_path)
    Path(config.checkpoint_path).unlink()
    app = web.create_app(
        web.WebConfig(
            checkpoint_path=str(tmp_path / "missing.pt"),
            checkpoint_glob=str(tmp_path / "model-step-*.pt"),
            manifest_path=config.manifest_path,
            display_size=4,
        )
    )

    state = _request(app, "GET", "/api/state")
    response = _request(app, "POST", "/api/generate", json_body={"index": 0})

    assert state.status_code == 200
    checkpoint = _checkpoint_payload(state.json(), "primary")
    assert checkpoint["path"] == ""
    assert checkpoint["available"] is False
    assert response.status_code == 503
    assert "No web app checkpoints match" in response.json()["detail"]


def test_web_generation_reloads_when_latest_checkpoint_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _write_web_fixture(tmp_path)
    first_checkpoint = tmp_path / "model-step-1.pt"
    next_checkpoint = tmp_path / "model-step-2.pt"
    first_checkpoint.write_bytes(b"first")
    load_calls: list[Path] = []

    def fake_load_generator(checkpoint_path: Path, *, device: str) -> SimpleNamespace:
        load_calls.append(checkpoint_path)
        return SimpleNamespace(checkpoint_path=checkpoint_path)

    def fake_generate(generator: SimpleNamespace, _: GenerationConfig) -> SimpleNamespace:
        return SimpleNamespace(image=object(), checkpoint_path=generator.checkpoint_path)

    monkeypatch.setattr(web, "load_generator", fake_load_generator)
    monkeypatch.setattr(web, "generate_image_with_model", fake_generate)
    monkeypatch.setattr(web, "_image_tensor_png", lambda image, display_size: b"png")
    app = web.create_app(
        web.WebConfig(
            checkpoint_path=config.checkpoint_path,
            checkpoint_glob=str(tmp_path / "model-step-*.pt"),
            manifest_path=config.manifest_path,
            display_size=4,
        )
    )

    first_response = _request(app, "POST", "/api/generate", json_body={"index": 0})
    next_checkpoint.write_bytes(b"next")
    second_response = _request(app, "POST", "/api/generate", json_body={"index": 0})

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert load_calls == [first_checkpoint, next_checkpoint]


def test_web_generation_reloads_when_checkpoint_file_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _write_web_fixture(tmp_path)
    checkpoint_path = Path(config.checkpoint_path)
    load_calls: list[Path] = []

    def fake_load_generator(checkpoint_path: Path, *, device: str) -> SimpleNamespace:
        load_calls.append(checkpoint_path)
        return SimpleNamespace(checkpoint_path=checkpoint_path)

    def fake_generate(generator: SimpleNamespace, _: GenerationConfig) -> SimpleNamespace:
        return SimpleNamespace(image=object(), checkpoint_path=generator.checkpoint_path)

    monkeypatch.setattr(web, "load_generator", fake_load_generator)
    monkeypatch.setattr(web, "generate_image_with_model", fake_generate)
    monkeypatch.setattr(web, "_image_tensor_png", lambda image, display_size: b"png")
    app = web.create_app(config)

    first_response = _request(app, "POST", "/api/generate", json_body={"index": 0})
    checkpoint_path.write_bytes(b"replacement checkpoint")
    second_response = _request(app, "POST", "/api/generate", json_body={"index": 0})

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert load_calls == [checkpoint_path, checkpoint_path]


def test_web_generate_stream_reports_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _write_web_fixture(tmp_path)

    def fake_generate(
        generator: object,
        generation_config: GenerationConfig,
        *,
        progress: Any,
    ) -> SimpleNamespace:
        assert generation_config.steps == 3
        for step in range(1, generation_config.steps + 1):
            progress(step, generation_config.steps)
        return SimpleNamespace(image=object())

    monkeypatch.setattr(web, "load_generator", _fake_load_generator)
    monkeypatch.setattr(web, "generate_image_with_model", fake_generate)
    monkeypatch.setattr(web, "_image_tensor_png", lambda image, display_size: b"png")
    app = web.create_app(config)
    assert any(getattr(route, "path", "") == "/api/generate/stream" for route in app.routes)
    prepared = web._prepare_generation(
        web.GenerateRequest(index=0, steps=3),
        web._load_samples(config),
        config,
    )

    events = [json.loads(line) for line in web._generate_stream(prepared, config, lambda: object())]
    assert [event["event"] for event in events] == [
        "start",
        "step",
        "step",
        "step",
        "complete",
    ]
    assert [(event["step"], event["steps"]) for event in events[1:4]] == [
        (1, 3),
        (2, 3),
        (3, 3),
    ]
    assert events[-1]["image"] == "data:image/png;base64,cG5n"
    assert events[-1]["sample"]["name"] == "tile"


def test_web_generate_rejects_bad_request_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _write_web_fixture(tmp_path)
    app = web.create_app(config)

    bad_index = _request(app, "POST", "/api/generate", json_body={"index": 99})
    assert bad_index.status_code == 400
    assert bad_index.json()["detail"] == "Sample index out of range: 99"

    bad_steps = _request(app, "POST", "/api/generate", json_body={"index": 0, "steps": 0})
    assert bad_steps.status_code == 400
    assert bad_steps.json()["detail"] == "steps must be positive"


def test_web_generate_uses_edited_mask_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, original_mask_path = _write_web_fixture(tmp_path)
    edited_mask = Image.new("RGB", (4, 4), OPENEARTHMAP_RGB["building"])
    received: dict[str, Path] = {}

    def fake_generate(generator: object, generation_config: GenerationConfig) -> SimpleNamespace:
        mask_path = Path(generation_config.mask_path)
        received["mask_path"] = mask_path
        assert mask_path != original_mask_path
        assert mask_path.exists()
        with Image.open(mask_path) as mask:
            building_id = OPENEARTHMAP_CLASSES.index("building")
            assert mask.mode == "L"
            assert list(_image_data(mask)) == [building_id] * 16
        return SimpleNamespace(image=object())

    monkeypatch.setattr(web, "load_generator", _fake_load_generator)
    monkeypatch.setattr(web, "generate_image_with_model", fake_generate)
    monkeypatch.setattr(web, "_image_tensor_png", lambda image, display_size: b"png")
    app = web.create_app(config)

    response = _request(
        app,
        "POST",
        "/api/generate",
        json_body={"index": 0, "maskImage": _png_data_uri(edited_mask)},
    )

    assert response.status_code == 200
    assert response.json()["maskEdited"] is True
    assert not received["mask_path"].exists()


def test_web_generate_rejects_invalid_edited_mask_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _write_web_fixture(tmp_path)
    app = web.create_app(config)

    response = _request(
        app,
        "POST",
        "/api/generate",
        json_body={"index": 0, "maskImage": "not-a-data-uri"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "maskImage must be a PNG data URI"


def test_web_generate_rejects_unmapped_edited_mask_color(
    tmp_path: Path,
) -> None:
    config, _ = _write_web_fixture(tmp_path)
    app = web.create_app(config)
    edited_mask = Image.new("RGB", (4, 4), (1, 2, 3))

    response = _request(
        app,
        "POST",
        "/api/generate",
        json_body={"index": 0, "maskImage": _png_data_uri(edited_mask)},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Mask contains unmapped RGB color: (1, 2, 3)"


def test_web_generate_rejects_wrong_size_edited_mask(
    tmp_path: Path,
) -> None:
    config, _ = _write_web_fixture(tmp_path)
    app = web.create_app(config)
    edited_mask = Image.new("RGB", (2, 2), OPENEARTHMAP_RGB["building"])

    response = _request(
        app,
        "POST",
        "/api/generate",
        json_body={"index": 0, "maskImage": _png_data_uri(edited_mask)},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "mask image must be 4x4 pixels"


def test_web_generate_rejects_oversized_mask_payload_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _write_web_fixture(tmp_path)

    def broken_decode(encoded: str, validate: bool) -> bytes:
        assert validate is True
        raise AssertionError("oversized payload should not be decoded")

    monkeypatch.setattr(web.base64, "b64decode", broken_decode)
    app = web.create_app(config)
    oversized_payload = "data:image/png;base64," + ("A" * 5000)

    response = _request(
        app,
        "POST",
        "/api/generate",
        json_body={"index": 0, "maskImage": oversized_payload},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "maskImage is too large"


def test_web_generate_keeps_server_errors_as_server_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _write_web_fixture(tmp_path)

    def fake_load_generator(checkpoint_path: str, *, device: str) -> object:
        return object()

    def broken_generate(generator: object, generation_config: GenerationConfig) -> SimpleNamespace:
        raise RuntimeError("bad checkpoint")

    monkeypatch.setattr(web, "load_generator", fake_load_generator)
    monkeypatch.setattr(web, "generate_image_with_model", broken_generate)
    app = web.create_app(config)

    response = _request(app, "POST", "/api/generate", json_body={"index": 0})

    assert response.status_code == 500


def test_web_create_app_rejects_missing_checkpoint(tmp_path: Path) -> None:
    config, _ = _write_web_fixture(tmp_path)
    Path(config.checkpoint_path).unlink()

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        web.create_app(config)


def test_web_create_app_rejects_missing_mask(tmp_path: Path) -> None:
    config, mask_path = _write_web_fixture(tmp_path)
    mask_path.unlink()

    with pytest.raises(FileNotFoundError, match="sample data not found"):
        web.create_app(config)


def test_web_uses_prepared_data_when_raw_files_are_not_mounted(tmp_path: Path) -> None:
    config, mask_path = _write_web_fixture(tmp_path)
    data_dir = Path(config.manifest_path).parent
    image_path = data_dir / "images" / "tile.png"
    prepared_path = data_dir / "prepared" / "tile.pt"
    write_prepared_sample(
        SatelliteSample(
            image_path=image_path,
            mask_path=mask_path,
            prepared_path=prepared_path,
            split="train",
        ),
        torch=torch,
        config=ModelConfig(mask_channels=len(OPENEARTHMAP_CLASSES)),
    )
    image_path.unlink()
    mask_path.unlink()

    app = web.create_app(config)

    mask = _request(app, "GET", "/api/mask", query_string="index=0")
    image = _request(app, "GET", "/api/image", query_string="index=0")

    assert mask.status_code == 200
    assert image.status_code == 200


def _write_web_fixture(tmp_path: Path) -> tuple[web.WebConfig, Path]:
    data_dir = tmp_path / "data" / "openearthmap"
    image_path = data_dir / "images" / "tile.png"
    mask_path = data_dir / "labels" / "tile.png"
    manifest_path = data_dir / "manifest.jsonl"
    checkpoint_path = tmp_path / "checkpoint.pt"

    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), (10, 20, 30)).save(image_path)
    Image.new("RGB", (2, 2), OPENEARTHMAP_RGB["unlabeled"]).save(mask_path)
    manifest_path.write_text(
        json.dumps(
            {
                "image": "images/tile.png",
                "mask": "labels/tile.png",
                "prepared": "prepared/tile.pt",
                "split": "train",
                "source": "openearthmap",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_path.write_bytes(b"checkpoint")

    return (
        web.WebConfig(
            checkpoint_path=str(checkpoint_path),
            manifest_path=str(manifest_path),
            display_size=4,
        ),
        mask_path,
    )


def _write_sam_proposals(path: Path, image_path: Path, mask_values: list[int]) -> None:
    mask_path = path.parent / "masks" / f"{image_path.stem}.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (2, 2))
    mask.putdata(mask_values)
    mask.save(mask_path)
    path.write_text(
        json.dumps(
            {
                "image_key": web.image_key(image_path),
                "mask": str(mask_path.relative_to(path.parent)),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _has_nontransparent_pixels(body: bytes) -> bool:
    with Image.open(BytesIO(body)) as image:
        rgba = image.convert("RGBA")
        return any(pixel[3] > 0 for pixel in _image_data(rgba))


def _fake_load_generator(checkpoint_path: str, *, device: str) -> object:
    return object()


def _checkpoint_payload(payload: dict[str, Any], checkpoint_id: str) -> dict[str, Any]:
    for checkpoint in payload["checkpoints"]:
        if checkpoint["id"] == checkpoint_id:
            return checkpoint
    raise AssertionError(f"Missing checkpoint payload: {checkpoint_id}")


def _png_data_uri(image: Image.Image) -> str:
    body = BytesIO()
    image.save(body, format="PNG")
    encoded = base64.b64encode(body.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_data(image: Image.Image) -> Any:
    get_flattened_data = getattr(image, "get_flattened_data", None)
    if get_flattened_data is not None:
        return get_flattened_data()
    return image.getdata()


def _request(
    app: Any,
    method: str,
    path: str,
    *,
    query_string: str = "",
    json_body: dict[str, Any] | None = None,
) -> SimpleNamespace:
    import asyncio

    async def call() -> SimpleNamespace:
        body = b""
        headers: list[tuple[bytes, bytes]] = []
        if json_body is not None:
            body = json.dumps(json_body).encode("utf-8")
            headers.append((b"content-type", b"application/json"))

        messages: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": query_string.encode("ascii"),
            "headers": headers,
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

        try:
            await app(scope, receive, send)
        except Exception:
            if not messages:
                raise

        start = next(message for message in messages if message["type"] == "http.response.start")
        content = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        )
        response_headers = {
            key.decode("latin1"): value.decode("latin1") for key, value in start["headers"]
        }

        return SimpleNamespace(
            status_code=start["status"],
            headers=response_headers,
            content=content,
            json=lambda: json.loads(content.decode("utf-8")),
        )

    return asyncio.run(call())
