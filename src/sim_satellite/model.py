"""Mask-conditioned flow-matching model primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, cast

import torch
from torch import nn

CHANNEL_MULTIPLIERS = (1, 2, 4, 4)
BLOCKS_PER_LEVEL = 2
DEEP_ATTENTION_DOWNSAMPLE_FACTOR = 4
MAX_FULL_ATTENTION_TOKENS = 4096


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 32
    image_channels: int = 3
    mask_channels: int = 9
    hidden_channels: int = 32
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.image_size < 8:
            raise ValueError("image_size must be at least 8")
        if self.image_channels < 1:
            raise ValueError("image_channels must be positive")
        if self.mask_channels < 1:
            raise ValueError("mask_channels must be positive")
        if self.hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ModelConfig:
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"Unknown model config fields: {', '.join(unknown)}")
        return cls(**{name: values[name] for name in names if name in values})


def build_model(config: ModelConfig | None = None) -> ConditionalUNet:
    config = config or ModelConfig()
    if config.image_size < 32:
        raise ValueError("unet architecture requires image_size of at least 32")
    if config.image_size % 8 != 0:
        raise ValueError("unet architecture requires image_size divisible by 8")
    deep_attention_size = config.image_size // DEEP_ATTENTION_DOWNSAMPLE_FACTOR
    if deep_attention_size * deep_attention_size > MAX_FULL_ATTENTION_TOKENS:
        raise ValueError(
            f"unet architecture limits deep full attention to {MAX_FULL_ATTENTION_TOKENS} "
            "tokens to avoid attention OOM"
        )
    return ConditionalUNet(config)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        half_channels = channels // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_channels).float()
            / max(1, half_channels - 1)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = timestep.float().view(-1)
        frequencies = cast(torch.Tensor, self.frequencies).to(device=timestep.device)
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
        self.time_projection = nn.Linear(time_channels, out_channels * 2)
        self.mask_projection = nn.Conv2d(mask_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        self.activation = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        semantic_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(self.activation(self.norm1(x)))
        h = h + self.mask_projection(semantic_mask)
        time_scale, time_shift = self.time_projection(time_embedding).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + time_scale[:, :, None, None]) + time_shift[:, :, None, None]
        h = self.conv2(self.activation(h))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.heads = _attention_head_count(channels)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        attended = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(-2, -1),
            key.transpose(-2, -1),
            value.transpose(-2, -1),
            dropout_p=0.0,
        )
        attended = attended.transpose(-2, -1).reshape(batch_size, channels, height, width)
        return x + self.projection(attended)


class ConditionalUNet(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        base_channels = model_config.hidden_channels
        time_channels = base_channels * 4
        levels = [base_channels * multiplier for multiplier in CHANNEL_MULTIPLIERS]

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.coordinate_projection = nn.Sequential(
            nn.Linear(3, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, base_channels),
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
                self.upsamples.append(nn.Conv2d(current_channels, current_channels, 3, padding=1))

        self.output_norm = nn.GroupNorm(_group_count(current_channels), current_channels)
        self.output_activation = nn.SiLU()
        self.output_conv = nn.Conv2d(
            current_channels,
            model_config.image_channels,
            3,
            padding=1,
        )

    def forward(
        self,
        image: torch.Tensor,
        semantic_mask: torch.Tensor,
        timestep: torch.Tensor,
        coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time_embedding = self.time_embedding(timestep)
        coordinate_bias = None
        if coordinates is not None:
            coordinate_features = _coordinate_features(coordinates)
            if coordinate_features.shape[0] != image.shape[0]:
                raise ValueError(
                    "Coordinate batch size must match image batch size: "
                    f"{coordinate_features.shape[0]} != {image.shape[0]}"
                )
            coordinate_bias = self.coordinate_projection(coordinate_features)
        mask_pyramid = {_spatial_size(semantic_mask): semantic_mask}
        h = self.input_conv(torch.cat([image, semantic_mask], dim=1))
        if coordinate_bias is not None:
            h = h + coordinate_bias[:, :, None, None]
        skips: list[torch.Tensor] = [h]

        for level_index, level_blocks_module in enumerate(self.down_blocks):
            level_blocks = cast(nn.ModuleList, level_blocks_module)
            for block_module in level_blocks:
                block = cast(ConditionedResBlock, block_module)
                h = block(
                    h,
                    time_embedding,
                    _semantic_mask_at(semantic_mask, _spatial_size(h), mask_pyramid),
                )
                skips.append(h)
            if level_index < len(self.downsamples):
                h = self.downsamples[level_index](h)
                skips.append(h)

        h = self.mid_block1(
            h,
            time_embedding,
            _semantic_mask_at(semantic_mask, _spatial_size(h), mask_pyramid),
        )
        h = self.mid_attention(h)
        h = self.mid_block2(
            h,
            time_embedding,
            _semantic_mask_at(semantic_mask, _spatial_size(h), mask_pyramid),
        )

        for level_index, level_blocks_module in enumerate(self.up_blocks):
            level_blocks = cast(nn.ModuleList, level_blocks_module)
            for block_module in level_blocks:
                block = cast(ConditionedResBlock, block_module)
                h = torch.cat([h, skips.pop()], dim=1)
                h = block(
                    h,
                    time_embedding,
                    _semantic_mask_at(semantic_mask, _spatial_size(h), mask_pyramid),
                )
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


def _spatial_size(tensor: torch.Tensor) -> tuple[int, int]:
    return int(tensor.shape[-2]), int(tensor.shape[-1])


def _coordinate_features(coordinates: torch.Tensor) -> torch.Tensor:
    coordinates = coordinates.float()
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f"Expected coordinates with shape (batch, 2), got {tuple(coordinates.shape)}"
        )

    longitude = coordinates[:, 0] * math.pi / 180.0
    latitude = coordinates[:, 1] * math.pi / 180.0
    return torch.stack(
        [
            torch.cos(latitude) * torch.cos(longitude),
            torch.cos(latitude) * torch.sin(longitude),
            torch.sin(latitude),
        ],
        dim=1,
    )


def _semantic_mask_at(
    semantic_mask: torch.Tensor,
    size: tuple[int, int],
    mask_pyramid: dict[tuple[int, int], torch.Tensor],
) -> torch.Tensor:
    if size not in mask_pyramid:
        mask_pyramid[size] = torch.nn.functional.interpolate(
            semantic_mask,
            size=size,
            mode="area",
        )
    return mask_pyramid[size]


def interpolate_flow_sample(
    clean_image: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    scale = timestep.float().view(-1, 1, 1, 1)
    return clean_image * (1.0 - scale) + noise * scale


def flow_velocity(clean_image: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return noise - clean_image
