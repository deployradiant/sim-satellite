"""Semantic segmentation tools for satellite imagery."""

from sim_satellite.segmentation.cli import evaluate_cli, predict_cli, train_cli
from sim_satellite.segmentation.config import (
    DEFAULT_SEGMENTATION_CHECKPOINT_PATH,
    DEFAULT_SEGMENTATION_PREDICTION_DIR,
    DEFAULT_SEGMENTATION_PREDICTION_TILE_STRIDE,
    SMP_UNET_EFFICIENTNET_B4_SCSE_MODEL_NAME,
    SegmentationEvaluationConfig,
    SegmentationEvaluationResult,
    SegmentationPredictionConfig,
    SegmentationPredictionResult,
    SegmentationTrainingConfig,
    SegmentationTrainingResult,
    load_segmentation_training_config_file,
)
from sim_satellite.segmentation.inference import (
    evaluate_segmentation_model,
    predict_segmentation_masks,
)
from sim_satellite.segmentation.models import build_segmentation_model
from sim_satellite.segmentation.training import train_segmentation_model

__all__ = [
    "DEFAULT_SEGMENTATION_CHECKPOINT_PATH",
    "DEFAULT_SEGMENTATION_PREDICTION_DIR",
    "DEFAULT_SEGMENTATION_PREDICTION_TILE_STRIDE",
    "SMP_UNET_EFFICIENTNET_B4_SCSE_MODEL_NAME",
    "SegmentationEvaluationConfig",
    "SegmentationEvaluationResult",
    "SegmentationPredictionConfig",
    "SegmentationPredictionResult",
    "SegmentationTrainingConfig",
    "SegmentationTrainingResult",
    "build_segmentation_model",
    "evaluate_cli",
    "evaluate_segmentation_model",
    "load_segmentation_training_config_file",
    "predict_cli",
    "predict_segmentation_masks",
    "train_cli",
    "train_segmentation_model",
]
