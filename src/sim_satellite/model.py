"""Mask-conditioned flow-matching model primitives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

CHANNEL_MULTIPLIERS = (1, 2, 4, 4)
BLOCKS_PER_LEVEL = 2


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 32
    image_channels: int = 3
    mask_channels: int = 9
    hidden_channels: int = 32
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.image_size < 8:
            raise ValueError("image_size must be at least 8")
        if self.image_channels < 1:
            raise ValueError("image_channels must be positive")
        if self.mask_channels < 1:
            raise ValueError("mask_channels must be positive")
        if self.hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ModelConfig:
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"Unknown model config fields: {', '.join(unknown)}")
        return cls(**{name: values[name] for name in names if name in values})


def require_torch() -> tuple[Any, Any]:
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required for model execution. Install the project "
            "runtime dependencies or run this through the Modal target."
        ) from _TORCH_IMPORT_ERROR

    return torch, nn


def build_model(config: ModelConfig | None = None) -> Any:
    require_torch()
    config = config or ModelConfig()
    _validate_unet_config(config)
    return ConditionalUNet(config)


def _validate_unet_config(config: ModelConfig) -> None:
    if config.image_size < 32:
        raise ValueError("unet architecture requires image_size of at least 32")
    if config.image_size % 8 != 0:
        raise ValueError("unet architecture requires image_size divisible by 8")


if nn is not None:

    class SinusoidalTimeEmbedding(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.channels = channels

        def forward(self, timestep: Any) -> Any:
            timestep = timestep.float().view(-1)
            half_channels = self.channels // 2
            frequencies = torch.exp(
                -torch.log(torch.tensor(10000.0, device=timestep.device))
                * torch.arange(half_channels, device=timestep.device).float()
                / max(1, half_channels - 1)
            )
            angles = timestep[:, None] * frequencies[None, :] * 1000.0
            embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
            if self.channels % 2:
                embedding = torch.nn.functional.pad(embedding, (0, 1))
            return embedding


    class ConditionedResBlock(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            time_channels: int,
            mask_channels: int,
        ) -> None:
            super().__init__()
            self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.time_projection = nn.Linear(time_channels, out_channels)
            self.mask_projection = nn.Conv2d(mask_channels, out_channels, 3, padding=1)
            self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv2d(in_channels, out_channels, 1)
            )
            self.activation = nn.SiLU()

        def forward(self, x: Any, time_embedding: Any, semantic_mask: Any) -> Any:
            h = self.conv1(self.activation(self.norm1(x)))
            h = h + self.time_projection(time_embedding)[:, :, None, None]
            resized_mask = torch.nn.functional.interpolate(
                semantic_mask,
                size=h.shape[-2:],
                mode="nearest",
            )
            h = h + self.mask_projection(resized_mask)
            h = self.conv2(self.activation(self.norm2(h)))
            return h + self.skip(x)


    class SelfAttention2d(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.heads = _attention_head_count(channels)
            self.norm = nn.GroupNorm(_group_count(channels), channels)
            self.qkv = nn.Conv2d(channels, channels * 3, 1)
            self.projection = nn.Conv2d(channels, channels, 1)

        def forward(self, x: Any) -> Any:
            batch_size, channels, height, width = x.shape
            head_channels = channels // self.heads
            normalized = self.norm(x)
            qkv = self.qkv(normalized).view(
                batch_size,
                3,
                self.heads,
                head_channels,
                height * width,
            )
            query, key, value = qkv.unbind(dim=1)
            scale = head_channels**-0.5
            attention = torch.einsum("bhcn,bhcm->bhnm", query * scale, key)
            attention = torch.softmax(attention, dim=-1)
            attended = torch.einsum("bhnm,bhcm->bhcn", attention, value)
            attended = attended.reshape(batch_size, channels, height, width)
            return x + self.projection(attended)


    class ConditionalUNet(nn.Module):
        def __init__(self, model_config: ModelConfig) -> None:
            super().__init__()
            self.config = model_config
            base_channels = model_config.hidden_channels
            time_channels = base_channels * 4
            levels = [base_channels * multiplier for multiplier in CHANNEL_MULTIPLIERS]

            self.time_mlp = nn.Sequential(
                SinusoidalTimeEmbedding(base_channels),
                nn.Linear(base_channels, time_channels),
                nn.SiLU(),
                nn.Linear(time_channels, time_channels),
            )
            self.input_conv = nn.Conv2d(
                model_config.image_channels + model_config.mask_channels,
                base_channels,
                3,
                padding=1,
            )

            current_channels = base_channels
            skip_channels: list[int] = [current_channels]
            self.down_blocks = nn.ModuleList()
            self.downsamples = nn.ModuleList()
            for level_index, level_channels in enumerate(levels):
                level_blocks = nn.ModuleList()
                for _ in range(BLOCKS_PER_LEVEL):
                    level_blocks.append(
                        ConditionedResBlock(
                            current_channels,
                            level_channels,
                            time_channels=time_channels,
                            mask_channels=model_config.mask_channels,
                        )
                    )
                    current_channels = level_channels
                    skip_channels.append(current_channels)
                self.down_blocks.append(level_blocks)
                if level_index != len(levels) - 1:
                    self.downsamples.append(nn.Conv2d(current_channels, current_channels, 3, 2, 1))
                    skip_channels.append(current_channels)

            self.mid_block1 = ConditionedResBlock(
                current_channels,
                current_channels,
                time_channels=time_channels,
                mask_channels=model_config.mask_channels,
            )
            self.mid_attention = SelfAttention2d(current_channels)
            self.mid_block2 = ConditionedResBlock(
                current_channels,
                current_channels,
                time_channels=time_channels,
                mask_channels=model_config.mask_channels,
            )

            self.up_blocks = nn.ModuleList()
            self.upsamples = nn.ModuleList()
            skip_channels_stack = list(skip_channels)
            for level_index, level_channels in reversed(list(enumerate(levels))):
                level_blocks = nn.ModuleList()
                for _ in range(BLOCKS_PER_LEVEL + 1):
                    skip_channel_count = skip_channels_stack.pop()
                    level_blocks.append(
                        ConditionedResBlock(
                            current_channels + skip_channel_count,
                            level_channels,
                            time_channels=time_channels,
                            mask_channels=model_config.mask_channels,
                        )
                    )
                    current_channels = level_channels
                self.up_blocks.append(level_blocks)
                if level_index != 0:
                    self.upsamples.append(
                        nn.Conv2d(current_channels, current_channels, 3, padding=1)
                    )

            self.output_norm = nn.GroupNorm(_group_count(current_channels), current_channels)
            self.output_activation = nn.SiLU()
            self.output_conv = nn.Conv2d(
                current_channels,
                model_config.image_channels,
                3,
                padding=1,
            )

        def forward(self, image: Any, semantic_mask: Any, timestep: Any) -> Any:
            time_embedding = self.time_mlp(timestep)
            h = self.input_conv(torch.cat([image, semantic_mask], dim=1))
            skips: list[Any] = [h]

            for level_index, level_blocks in enumerate(self.down_blocks):
                for block in level_blocks:
                    h = block(h, time_embedding, semantic_mask)
                    skips.append(h)
                if level_index < len(self.downsamples):
                    h = self.downsamples[level_index](h)
                    skips.append(h)

            h = self.mid_block1(h, time_embedding, semantic_mask)
            h = self.mid_attention(h)
            h = self.mid_block2(h, time_embedding, semantic_mask)

            for level_index, level_blocks in enumerate(self.up_blocks):
                for block in level_blocks:
                    h = torch.cat([h, skips.pop()], dim=1)
                    h = block(h, time_embedding, semantic_mask)
                if level_index < len(self.upsamples):
                    h = torch.nn.functional.interpolate(h, scale_factor=2, mode="nearest")
                    h = self.upsamples[level_index](h)

            h = self.output_activation(self.output_norm(h))
            return self.output_conv(h)


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _attention_head_count(channels: int) -> int:
    preferred = max(1, channels // 64)
    for heads in range(preferred, 0, -1):
        if channels % heads == 0:
            return heads
    return 1


def interpolate_flow_sample(clean_image: Any, noise: Any, timestep: Any) -> Any:
    scale = timestep.float().view(-1, 1, 1, 1)
    return clean_image * (1.0 - scale) + noise * scale


def flow_velocity(clean_image: Any, noise: Any) -> Any:
    return noise - clean_image
