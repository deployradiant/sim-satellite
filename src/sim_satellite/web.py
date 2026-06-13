"""Local web app for mask-conditioned generation."""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image

from sim_satellite.data.loader import SatelliteDataset, SatelliteSample
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES, OPENEARTHMAP_RGB
from sim_satellite.data.tensors import mask_to_class_ids
from sim_satellite.infer import InferenceConfig, generate_image
from sim_satellite.model import ModelConfig
from sim_satellite.train import DEFAULT_MANIFEST_PATH

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_DISPLAY_SIZE = 256
DEFAULT_MAX_SAMPLES = 32
DEFAULT_WEB_CHECKPOINT_PATH = "artifacts/tiny-sim-satellite-modal.pt"


@dataclass(frozen=True)
class WebConfig:
    checkpoint_path: str = DEFAULT_WEB_CHECKPOINT_PATH
    manifest_path: str = DEFAULT_MANIFEST_PATH
    split: str = "train"
    device: str = "cpu"
    steps: int = 8
    seed: int = 1337
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_samples: int = DEFAULT_MAX_SAMPLES
    display_size: int = DEFAULT_DISPLAY_SIZE

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.port < 1:
            raise ValueError("port must be positive")
        if self.max_samples < 1:
            raise ValueError("max_samples must be positive")
        if self.display_size < 1:
            raise ValueError("display_size must be positive")


def serve(config: WebConfig | None = None) -> None:
    config = config or WebConfig()
    samples = _load_samples(config)
    handler = _make_handler(config, samples)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    print(f"Serving sim-satellite at http://{config.host}:{config.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _load_samples(config: WebConfig) -> tuple[SatelliteSample, ...]:
    manifest_path = Path(config.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Web app manifest not found: {manifest_path}. "
            "Run `sim-satellite data openearthmap` first."
        )
    samples = tuple(SatelliteDataset(manifest_path, split=config.split))
    if not samples:
        raise RuntimeError(f"No samples for split `{config.split}` in {manifest_path}")
    return samples[: config.max_samples]


def _make_handler(
    config: WebConfig,
    samples: Sequence[SatelliteSample],
) -> type[BaseHTTPRequestHandler]:
    class SimSatelliteHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            route = urlparse(self.path)
            if route.path == "/":
                self._send_html(_render_html(config))
                return
            if route.path == "/api/state":
                self._send_json(_state_payload(config, samples))
                return
            if route.path == "/api/mask":
                try:
                    index = _query_int(route.query, "index", 0)
                    mask = _mask_preview_png(_sample_at(samples, index).mask_path, config)
                except ValueError as exc:
                    self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                    return
                self._send_bytes(mask, "image/png")
                return
            if route.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            route = urlparse(self.path)
            if route.path != "/api/generate":
                self._send_error_json(HTTPStatus.NOT_FOUND, "Not found")
                return

            try:
                payload = self._read_json()
                index = int(payload.get("index", 0))
                sample = _sample_at(samples, index)
                seed = int(payload.get("seed", config.seed))
                steps = int(payload.get("steps", config.steps))
                if steps < 1:
                    raise ValueError("steps must be positive")
                generated = generate_image(
                    InferenceConfig(
                        seed=seed,
                        checkpoint_path=config.checkpoint_path,
                        output_path="artifacts/web-sample.pt",
                        manifest_path=config.manifest_path,
                        split=config.split,
                        mask_path=str(sample.mask_path),
                        device=config.device,
                        steps=steps,
                    )
                )
                png = _image_tensor_png(generated.image, config.display_size)
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return

            self._send_json(
                {
                    "image": _data_uri(png, "image/png"),
                    "sample": _sample_payload(index, sample),
                    "seed": seed,
                    "steps": steps,
                }
            )

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def _send_html(self, html: str) -> None:
            self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")

        def _send_json(self, payload: dict[str, Any]) -> None:
            self._send_bytes(
                json.dumps(payload).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _send_error_json(self, status: HTTPStatus, message: str) -> None:
            body = json.dumps({"error": message}).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return SimSatelliteHandler


def _state_payload(config: WebConfig, samples: Sequence[SatelliteSample]) -> dict[str, Any]:
    return {
        "checkpointPath": config.checkpoint_path,
        "manifestPath": config.manifest_path,
        "split": config.split,
        "device": config.device,
        "steps": config.steps,
        "seed": config.seed,
        "samples": [_sample_payload(index, sample) for index, sample in enumerate(samples)],
        "classes": [
            {
                "id": index,
                "name": name,
                "rgb": OPENEARTHMAP_RGB[name],
            }
            for index, name in enumerate(OPENEARTHMAP_CLASSES)
        ],
    }


def _sample_payload(index: int, sample: SatelliteSample) -> dict[str, Any]:
    return {
        "index": index,
        "name": sample.image_path.stem,
        "imagePath": str(sample.image_path),
        "maskPath": str(sample.mask_path),
        "source": sample.source,
        "split": sample.split,
    }


def _sample_at(samples: Sequence[SatelliteSample], index: int) -> SatelliteSample:
    if index < 0 or index >= len(samples):
        raise ValueError(f"Sample index out of range: {index}")
    return samples[index]


def _mask_preview_png(mask_path: Path, config: WebConfig) -> bytes:
    model_config = ModelConfig(mask_channels=len(OPENEARTHMAP_CLASSES))
    with Image.open(mask_path) as mask:
        resized = mask.resize(
            (config.display_size, config.display_size),
            Image.Resampling.NEAREST,
        )
        labels = mask_to_class_ids(resized, model_config)

    image = Image.new("RGB", (config.display_size, config.display_size))
    colors = [OPENEARTHMAP_RGB[name] for name in OPENEARTHMAP_CLASSES]
    image.putdata([colors[label] for label in labels])
    return _pil_png(image)


def _image_tensor_png(image: Any, display_size: int) -> bytes:
    tensor = image.detach().cpu()[0].clamp(0.0, 1.0).mul(255).byte().permute(1, 2, 0)
    height = int(tensor.shape[0])
    width = int(tensor.shape[1])
    pixels = [tuple(int(channel) for channel in pixel) for row in tensor.tolist() for pixel in row]
    pil_image = Image.new("RGB", (width, height))
    pil_image.putdata(pixels)
    if width != display_size or height != display_size:
        pil_image = pil_image.resize((display_size, display_size), Image.Resampling.NEAREST)
    return _pil_png(pil_image)


def _pil_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _data_uri(body: bytes, content_type: str) -> str:
    encoded = base64.b64encode(body).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def _query_int(query: str, name: str, default: int) -> int:
    values = parse_qs(query)
    if name not in values:
        return default
    return int(values[name][0])


def _render_html(config: WebConfig) -> str:
    state = {
        "seed": config.seed,
        "steps": config.steps,
    }
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sim Satellite</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1d2520;
      --muted: #69736c;
      --line: #cfd8d1;
      --panel: #f7faf7;
      --panel-strong: #edf3ef;
      --accent: #2d6a4f;
      --accent-strong: #1b4332;
      --warn: #b84a20;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #fbfcfb;
    }}
    .shell {{
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto auto 1fr auto;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 22px 14px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      line-height: 1.1;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
      text-align: right;
      overflow-wrap: anywhere;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(180px, 2fr) repeat(2, minmax(84px, 120px)) auto;
      gap: 10px;
      align-items: end;
      padding: 14px 22px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    label {{
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}
    select,
    input,
    button {{
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font: inherit;
      font-size: 14px;
      background: #ffffff;
      color: var(--ink);
    }}
    select,
    input {{
      padding: 0 10px;
      width: 100%;
    }}
    button {{
      padding: 0 16px;
      border-color: var(--accent);
      background: var(--accent);
      color: #ffffff;
      font-weight: 700;
      cursor: pointer;
    }}
    button:hover {{
      background: var(--accent-strong);
    }}
    button:disabled {{
      cursor: wait;
      opacity: 0.65;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) 260px;
      gap: 18px;
      padding: 22px;
      align-items: start;
    }}
    .viewer {{
      min-width: 0;
      display: grid;
      gap: 10px;
    }}
    .viewer h2,
    .classes h2 {{
      margin: 0;
      font-size: 14px;
      line-height: 1.2;
      font-weight: 750;
      letter-spacing: 0;
    }}
    .frame {{
      aspect-ratio: 1;
      width: 100%;
      max-height: calc(100vh - 210px);
      min-height: 260px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
      display: grid;
      place-items: center;
      overflow: hidden;
    }}
    .frame img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      image-rendering: pixelated;
    }}
    .empty {{
      color: var(--muted);
      font-size: 13px;
    }}
    .classes {{
      display: grid;
      gap: 10px;
      padding-left: 2px;
    }}
    .class-list {{
      display: grid;
      gap: 7px;
    }}
    .class-row {{
      display: grid;
      grid-template-columns: 16px 1fr auto;
      gap: 8px;
      align-items: center;
      min-height: 24px;
      font-size: 13px;
    }}
    .swatch {{
      width: 16px;
      height: 16px;
      border-radius: 3px;
      border: 1px solid rgba(0, 0, 0, 0.18);
    }}
    .class-id {{
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }}
    footer {{
      min-height: 34px;
      padding: 8px 22px 12px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      background: #ffffff;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    footer.error {{
      color: var(--warn);
    }}
    @media (max-width: 960px) {{
      main {{
        grid-template-columns: 1fr;
      }}
      .frame {{
        max-height: none;
      }}
      .controls {{
        grid-template-columns: 1fr 1fr;
      }}
      button {{
        grid-column: span 2;
      }}
      header {{
        align-items: flex-start;
        flex-direction: column;
      }}
      .meta {{
        text-align: left;
      }}
    }}
    @media (max-width: 560px) {{
      .controls {{
        grid-template-columns: 1fr;
      }}
      button {{
        grid-column: auto;
      }}
      main,
      header,
      .controls,
      footer {{
        padding-left: 14px;
        padding-right: 14px;
      }}
      .frame {{
        min-height: 220px;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>Sim Satellite</h1>
      <div class="meta" id="meta">Loading checkpoint...</div>
    </header>
    <div class="controls">
      <label>
        Mask sample
        <select id="sample"></select>
      </label>
      <label>
        Seed
        <input id="seed" type="number" min="0" step="1">
      </label>
      <label>
        Steps
        <input id="steps" type="number" min="1" step="1">
      </label>
      <button id="generate" type="button">Generate</button>
    </div>
    <main>
      <section class="viewer">
        <h2>Condition mask</h2>
        <div class="frame"><img id="mask" alt="Condition mask"></div>
      </section>
      <section class="viewer">
        <h2>Generated image</h2>
        <div class="frame" id="output-frame"><div class="empty">No image yet</div></div>
      </section>
      <aside class="classes">
        <h2>Classes</h2>
        <div id="classes" class="class-list"></div>
      </aside>
    </main>
    <footer id="status">Ready</footer>
  </div>
  <script>
    const initialState = {json.dumps(state)};
    const sampleSelect = document.getElementById("sample");
    const seedInput = document.getElementById("seed");
    const stepsInput = document.getElementById("steps");
    const generateButton = document.getElementById("generate");
    const maskImage = document.getElementById("mask");
    const outputFrame = document.getElementById("output-frame");
    const statusEl = document.getElementById("status");
    const classesEl = document.getElementById("classes");
    const metaEl = document.getElementById("meta");

    seedInput.value = initialState.seed;
    stepsInput.value = initialState.steps;

    function setStatus(text, isError = false) {{
      statusEl.textContent = text;
      statusEl.classList.toggle("error", isError);
    }}

    function renderMask(index) {{
      maskImage.src = `/api/mask?index=${{index}}&t=${{Date.now()}}`;
    }}

    async function loadState() {{
      const response = await fetch("/api/state");
      const state = await response.json();
      if (!response.ok) throw new Error(state.error || "Could not load state");

      metaEl.textContent = [
        state.checkpointPath,
        state.device,
        `${{state.samples.length}} masks`
      ].join(" | ");
      sampleSelect.innerHTML = "";
      state.samples.forEach((sample) => {{
        const option = document.createElement("option");
        option.value = sample.index;
        option.textContent = `${{sample.index}} - ${{sample.name}}`;
        sampleSelect.appendChild(option);
      }});
      classesEl.innerHTML = "";
      state.classes.forEach((item) => {{
        const row = document.createElement("div");
        row.className = "class-row";
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.backgroundColor = `rgb(${{item.rgb[0]}}, ${{item.rgb[1]}}, ${{item.rgb[2]}})`;
        const name = document.createElement("span");
        name.textContent = item.name;
        const id = document.createElement("span");
        id.className = "class-id";
        id.textContent = item.id;
        row.append(swatch, name, id);
        classesEl.appendChild(row);
      }});
      renderMask(sampleSelect.value || 0);
    }}

    async function generate() {{
      generateButton.disabled = true;
      setStatus("Generating...");
      try {{
        const response = await fetch("/api/generate", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{
            index: Number(sampleSelect.value),
            seed: Number(seedInput.value),
            steps: Number(stepsInput.value)
          }})
        }});
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Generation failed");
        const image = document.createElement("img");
        image.id = "generated";
        image.alt = "Generated satellite image";
        image.src = payload.image;
        outputFrame.replaceChildren(image);
        const message = [
          `Generated ${{payload.sample.name}}`,
          `seed ${{payload.seed}}`,
          `${{payload.steps}} steps`
        ].join(" | ");
        setStatus(message);
      }} catch (error) {{
        setStatus(error.message, true);
      }} finally {{
        generateButton.disabled = false;
      }}
    }}

    sampleSelect.addEventListener("change", () => renderMask(sampleSelect.value));
    generateButton.addEventListener("click", generate);
    loadState().catch((error) => setStatus(error.message, true));
  </script>
</body>
</html>
"""


def _build_click_command() -> Any:
    try:
        import click
    except ModuleNotFoundError:
        return None

    @click.command("app", help="Launch the local mask-conditioned generation app.")
    @click.option("--checkpoint-path", default=WebConfig.checkpoint_path, show_default=True)
    @click.option("--manifest-path", default=WebConfig.manifest_path, show_default=True)
    @click.option("--split", default=WebConfig.split, show_default=True)
    @click.option("--device", default=WebConfig.device, show_default=True)
    @click.option("--steps", default=WebConfig.steps, show_default=True)
    @click.option("--seed", default=WebConfig.seed, show_default=True)
    @click.option("--host", default=WebConfig.host, show_default=True)
    @click.option("--port", default=WebConfig.port, show_default=True, type=int)
    @click.option("--max-samples", default=WebConfig.max_samples, show_default=True)
    def command(
        checkpoint_path: str,
        manifest_path: str,
        split: str,
        device: str,
        steps: int,
        seed: int,
        host: str,
        port: int,
        max_samples: int,
    ) -> None:
        serve(
            WebConfig(
                checkpoint_path=checkpoint_path,
                manifest_path=manifest_path,
                split=split,
                device=device,
                steps=steps,
                seed=seed,
                host=host,
                port=port,
                max_samples=max_samples,
            )
        )

    return command


cli = _build_click_command()


if __name__ == "__main__":
    if cli is None:
        raise SystemExit("Click is required for the web app CLI.")
    cli()
