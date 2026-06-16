"""Training run logging, metrics, plots, and checkpoint artifacts."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sim_satellite.model import ModelConfig


def print_train_start(
    *,
    started_at_epoch_utc: float,
    start_step: int,
    steps: int,
    train_samples: int,
    train_batches: int,
    batch_size: int,
    timestep_sampling: str,
    timestep_logit_normal_mean: float,
    timestep_logit_normal_std: float,
    timestep_uniform_mix_probability: float,
    precision: str,
    preload_sample_count: int,
    preload_seconds: float,
    timing_sample_every: int,
    model_parameters: int,
    image_size: int,
    device: Any,
    torch_device_payload: Mapping[str, int | str | None],
) -> None:
    _print_event(
        {
            "event": "train_start",
            "started_at_epoch_utc": started_at_epoch_utc,
            "start_step": start_step,
            "steps": steps,
            "remaining_steps": max(steps - start_step, 0),
            "train_samples": train_samples,
            "train_batches": train_batches,
            "batch_size": batch_size,
            "total_images_target": steps * batch_size,
            "timestep_sampling": timestep_sampling,
            "timestep_logit_normal_mean": timestep_logit_normal_mean,
            "timestep_logit_normal_std": timestep_logit_normal_std,
            "timestep_uniform_mix_probability": timestep_uniform_mix_probability,
            "precision": precision,
            "preload_sample_count": preload_sample_count,
            "preload_seconds": preload_seconds,
            "timing_sample_every": timing_sample_every,
            "model_parameters": model_parameters,
            "image_size": image_size,
            "device": str(device),
            **torch_device_payload,
        }
    )


def print_preload_start(
    *,
    manifest_path: str,
    split: str,
    sample_count: int,
    train_samples: int,
    validation_batches: int,
) -> None:
    _print_event(
        {
            "event": "preload_start",
            "manifest_path": manifest_path,
            "split": split,
            "sample_count": sample_count,
            "train_samples": train_samples,
            "validation_batches": validation_batches,
            "logged_at_epoch_utc": time.time(),
        }
    )


def print_preload_progress(
    *,
    started_at_monotonic: float,
    completed: int,
    total: int,
) -> None:
    now_monotonic = time.monotonic()
    elapsed_seconds = max(0.0, now_monotonic - started_at_monotonic)
    samples_per_second = completed / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    remaining = max(total - completed, 0)
    estimated_remaining_seconds = (
        remaining / samples_per_second if samples_per_second > 0.0 else None
    )
    _print_event(
        {
            "event": "preload_progress",
            "completed_samples": completed,
            "total_samples": total,
            "remaining_samples": remaining,
            "progress_fraction": completed / total if total > 0 else 0.0,
            "elapsed_seconds": elapsed_seconds,
            "samples_per_second": samples_per_second,
            "estimated_remaining_seconds": estimated_remaining_seconds,
            "logged_at_epoch_utc": time.time(),
        }
    )


def print_validation_progress(
    *,
    started_at_epoch_utc: float,
    started_at_monotonic: float,
    start_step: int,
    step: int,
    steps: int,
    batch_size: int,
    loss: float,
) -> None:
    _print_event(
        {
            "event": "validation_progress",
            "step": step,
            "steps": steps,
            "loss": loss,
            **run_timing_payload(
                started_at_epoch_utc=started_at_epoch_utc,
                started_at_monotonic=started_at_monotonic,
                start_step=start_step,
                step=step,
                steps=steps,
                batch_size=batch_size,
            ),
        }
    )


def print_train_progress(
    *,
    started_at_epoch_utc: float,
    started_at_monotonic: float,
    start_step: int,
    step: int,
    steps: int,
    batch_size: int,
    loss: float,
    loss_record: Mapping[str, float | int],
    step_timing: Mapping[str, float | int],
) -> None:
    _print_event(
        {
            "event": "train_progress",
            "step": step,
            "steps": steps,
            "loss": loss,
            "step_seconds": loss_record["step_seconds"],
            "step_images_per_second": loss_record["step_images_per_second"],
            **step_timing,
            **run_timing_payload(
                started_at_epoch_utc=started_at_epoch_utc,
                started_at_monotonic=started_at_monotonic,
                start_step=start_step,
                step=step,
                steps=steps,
                batch_size=batch_size,
            ),
        }
    )


def run_timing_payload(
    *,
    started_at_epoch_utc: float,
    started_at_monotonic: float,
    start_step: int,
    step: int,
    steps: int,
    batch_size: int,
    now_epoch_utc: float | None = None,
    now_monotonic: float | None = None,
) -> dict[str, float | int | None]:
    now_epoch_utc = time.time() if now_epoch_utc is None else now_epoch_utc
    now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    elapsed_seconds = max(0.0, now_monotonic - started_at_monotonic)
    completed_steps_this_run = max(0, step - start_step)
    remaining_steps = max(0, steps - step)
    completed_images_this_run = completed_steps_this_run * batch_size
    remaining_images = remaining_steps * batch_size
    steps_per_second = (
        completed_steps_this_run / elapsed_seconds
        if elapsed_seconds > 0.0 and completed_steps_this_run > 0
        else 0.0
    )
    estimated_remaining_seconds = (
        remaining_steps / steps_per_second if steps_per_second > 0.0 else None
    )
    return {
        "started_at_epoch_utc": started_at_epoch_utc,
        "logged_at_epoch_utc": now_epoch_utc,
        "elapsed_seconds": elapsed_seconds,
        "completed_steps_this_run": completed_steps_this_run,
        "completed_images_this_run": completed_images_this_run,
        "remaining_steps": remaining_steps,
        "remaining_images": remaining_images,
        "progress_fraction": step / steps if steps > 0 else 0.0,
        "steps_per_second": steps_per_second,
        "images_per_second": steps_per_second * batch_size,
        "estimated_remaining_seconds": estimated_remaining_seconds,
        "estimated_total_seconds": (
            elapsed_seconds + estimated_remaining_seconds
            if estimated_remaining_seconds is not None
            else None
        ),
        "estimated_finish_epoch_utc": (
            now_epoch_utc + estimated_remaining_seconds
            if estimated_remaining_seconds is not None
            else None
        ),
    }


def torch_device_payload(torch: Any, device: Any) -> dict[str, int | str | None]:
    cuda = torch.cuda
    return {
        "torch_cuda_device_count": int(cuda.device_count()),
        "torch_device_name": str(cuda.get_device_name(device)),
    }


def loss_scalar(loss: Any) -> float:
    return float(loss.detach().cpu().item())


def loss_record(
    *,
    step: int,
    loss: float,
    step_seconds: float,
    batch_size: int,
    step_timing: Mapping[str, float | int],
) -> dict[str, float | int]:
    record: dict[str, float | int] = {
        "step": step,
        "loss": loss,
        "step_seconds": step_seconds,
        "step_images_per_second": batch_size / step_seconds if step_seconds > 0.0 else 0.0,
    }
    record.update(step_timing)
    return record


def write_training_artifacts(
    checkpoint_path: Path,
    metrics_path: Path,
    loss_plot_path: Path,
    *,
    model: Any,
    optimizer: Any,
    generator: Any,
    device_generator: Any | None,
    model_config: ModelConfig,
    config: Any,
    model_parameters: int,
    losses: list[dict[str, float | int]],
    validation_losses: list[dict[str, float | int]],
    final_loss: float,
    step: int,
    run_timing: Mapping[str, float | int | None],
    torch: Any,
) -> None:
    _write_training_metrics(
        metrics_path,
        model_config=model_config,
        config=config,
        model_parameters=model_parameters,
        losses=losses,
        validation_losses=validation_losses,
        final_loss=final_loss,
        step=step,
        run_timing=run_timing,
    )
    _write_loss_plot(loss_plot_path, losses, validation_losses)
    _write_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        generator=generator,
        device_generator=device_generator,
        model_config=model_config,
        config=config,
        losses=losses,
        validation_losses=validation_losses,
        final_loss=final_loss,
        step=step,
        metrics_path=metrics_path,
        loss_plot_path=loss_plot_path,
        torch=torch,
    )


def _print_event(payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _write_training_metrics(
    path: Path,
    *,
    model_config: ModelConfig,
    config: Any,
    model_parameters: int,
    losses: list[dict[str, float | int]],
    validation_losses: list[dict[str, float | int]],
    final_loss: float,
    step: int,
    run_timing: Mapping[str, float | int | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model_config": asdict(model_config),
                "training_config": asdict(config),
                "model_parameters": model_parameters,
                "step": step,
                "final_loss": final_loss,
                "run_timing": dict(run_timing),
                "losses": losses,
                "validation_losses": validation_losses,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_loss_plot(
    path: Path,
    losses: list[dict[str, float | int]],
    validation_losses: list[dict[str, float | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    steps = [int(item["step"]) for item in losses]
    values = [float(item["loss"]) for item in losses]
    smoothed = _exponential_moving_average(values, decay=0.95)

    figure, axis = pyplot.subplots(figsize=(8, 4.5), dpi=140)
    if steps:
        axis.plot(steps, values, color="#94a3b8", linewidth=1.0, alpha=0.55, label="train step")
        axis.plot(steps, smoothed, color="#0f766e", linewidth=2.0, label="train EMA 0.95")
    if validation_losses:
        validation_steps = [int(item["step"]) for item in validation_losses]
        validation_values = [float(item["loss"]) for item in validation_losses]
        axis.plot(
            validation_steps,
            validation_values,
            color="#dc2626",
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="validation",
        )
    axis.set_title("Training Loss")
    axis.set_xlabel("Step")
    axis.set_ylabel("MSE")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    pyplot.close(figure)


def _exponential_moving_average(values: list[float], *, decay: float) -> list[float]:
    if not values:
        return []
    average = values[0]
    smoothed = []
    for value in values:
        average = decay * average + (1.0 - decay) * value
        smoothed.append(average)
    return smoothed


def _write_training_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    generator: Any,
    device_generator: Any | None,
    model_config: ModelConfig,
    config: Any,
    losses: list[dict[str, float | int]],
    validation_losses: list[dict[str, float | int]],
    final_loss: float,
    step: int,
    metrics_path: Path,
    loss_plot_path: Path,
    torch: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "device_generator_state": (
                device_generator.get_state() if device_generator is not None else None
            ),
            "model_config": asdict(model_config),
            "training_config": asdict(config),
            "step": step,
            "losses": losses,
            "validation_losses": validation_losses,
            "final_loss": final_loss,
            "metrics_path": str(metrics_path),
            "loss_plot_path": str(loss_plot_path),
        },
        path,
    )
