from __future__ import annotations

import base64
import json
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim_satellite import web  # noqa: E402
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES, OPENEARTHMAP_RGB  # noqa: E402
from sim_satellite.infer import GenerationConfig  # noqa: E402


def test_web_state_and_mask_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _ = _write_web_fixture(tmp_path)
    app = web.create_app(config)

    state = _request(app, "GET", "/api/state")
    assert state.status_code == 200
    payload = state.json()
    assert payload["checkpointPath"] == config.checkpoint_path
    assert payload["samples"][0]["name"] == "tile"
    assert payload["classes"][0]["name"] == "unlabeled"

    mask = _request(app, "GET", "/api/mask", query_string="index=0")
    assert mask.status_code == 200
    assert mask.headers["content-type"] == "image/png"
    assert mask.content.startswith(b"\x89PNG")


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


def test_web_generate_uses_generation_config_without_output_path(
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
    assert not hasattr(generation_config, "output_path")
    assert not hasattr(generation_config, "manifest_path")
    assert not hasattr(generation_config, "split")
    assert generation_config.seed == 42
    assert generation_config.steps == 2
    assert generation_config.mask_path == str(mask_path)
    assert load_calls == [(config.checkpoint_path, config.device)]


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

    with pytest.raises(FileNotFoundError, match="mask not found"):
        web.create_app(config)


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


def _fake_load_generator(checkpoint_path: str, *, device: str) -> object:
    return object()


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
