"""Semantic segmentation tools for satellite imagery."""

from sim_satellite.segmentation.cli import evaluate_cli, predict_cli, prepare_cli, train_cli
from sim_satellite.segmentation.config import (
    DEFAULT_SEGMENTATION_CHECKPOINT_PATH,
    DEFAULT_SEGMENTATION_PREDICTION_DIR,
    DEFAULT_SEGMENTATION_VAE_CHECKPOINT_PATH,
    VAE_ASPP_UNET_MODEL_NAME,
    SegmentationEvaluationConfig,
    SegmentationEvaluationResult,
    SegmentationPredictionConfig,
    SegmentationPredictionResult,
    SegmentationPreparationConfig,
    SegmentationPreparationResult,
    SegmentationTrainingConfig,
    SegmentationTrainingResult,
    load_segmentation_training_config_file,
)
from sim_satellite.segmentation.inference import (
    evaluate_segmentation_model,
    predict_segmentation_masks,
)
from sim_satellite.segmentation.models import (
    VAEASPPUNetSegmentationModel,
    build_segmentation_model,
)
from sim_satellite.segmentation.preparation import prepare_segmentation_manifest
from sim_satellite.segmentation.training import train_segmentation_model

__all__ = [
    "DEFAULT_SEGMENTATION_CHECKPOINT_PATH",
    "DEFAULT_SEGMENTATION_PREDICTION_DIR",
    "DEFAULT_SEGMENTATION_VAE_CHECKPOINT_PATH",
    "VAE_ASPP_UNET_MODEL_NAME",
    "VAEASPPUNetSegmentationModel",
    "SegmentationEvaluationConfig",
    "SegmentationEvaluationResult",
    "SegmentationPredictionConfig",
    "SegmentationPredictionResult",
    "SegmentationPreparationConfig",
    "SegmentationPreparationResult",
    "SegmentationTrainingConfig",
    "SegmentationTrainingResult",
    "build_segmentation_model",
    "evaluate_cli",
    "evaluate_segmentation_model",
    "load_segmentation_training_config_file",
    "predict_cli",
    "predict_segmentation_masks",
    "prepare_cli",
    "prepare_segmentation_manifest",
    "train_cli",
    "train_segmentation_model",
]
