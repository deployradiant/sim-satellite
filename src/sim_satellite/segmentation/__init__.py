"""Semantic segmentation tools for satellite imagery."""

from sim_satellite.segmentation.cli import (
    evaluate_cli,
    predict_cli,
    pseudolabel_study_cli,
    train_cli,
)
from sim_satellite.segmentation.config import (
    DEFAULT_SEGMENTATION_CHECKPOINT_PATH,
    DEFAULT_SEGMENTATION_PREDICTION_DIR,
    DEFAULT_SEGMENTATION_VAE_CHECKPOINT_PATH,
    VAE_ASPP_UNET_MODEL_NAME,
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
from sim_satellite.segmentation.models import (
    VAEASPPUNetSegmentationModel,
    build_segmentation_model,
)
from sim_satellite.segmentation.pseudolabel import (
    OpenEarthMapPseudoComparisonManifests,
    OpenEarthMapPseudoLabelStudyConfig,
    OpenEarthMapPseudoLabelStudyResult,
    load_openearthmap_pseudolabel_study_config_file,
    run_openearthmap_pseudolabel_study,
    write_openearthmap_pseudolabel_comparison_manifests,
)
from sim_satellite.segmentation.training import train_segmentation_model

__all__ = [
    "DEFAULT_SEGMENTATION_CHECKPOINT_PATH",
    "DEFAULT_SEGMENTATION_PREDICTION_DIR",
    "DEFAULT_SEGMENTATION_VAE_CHECKPOINT_PATH",
    "VAE_ASPP_UNET_MODEL_NAME",
    "VAEASPPUNetSegmentationModel",
    "OpenEarthMapPseudoComparisonManifests",
    "OpenEarthMapPseudoLabelStudyConfig",
    "OpenEarthMapPseudoLabelStudyResult",
    "SegmentationEvaluationConfig",
    "SegmentationEvaluationResult",
    "SegmentationPredictionConfig",
    "SegmentationPredictionResult",
    "SegmentationTrainingConfig",
    "SegmentationTrainingResult",
    "build_segmentation_model",
    "evaluate_cli",
    "evaluate_segmentation_model",
    "load_openearthmap_pseudolabel_study_config_file",
    "load_segmentation_training_config_file",
    "predict_cli",
    "predict_segmentation_masks",
    "pseudolabel_study_cli",
    "run_openearthmap_pseudolabel_study",
    "train_cli",
    "train_segmentation_model",
    "write_openearthmap_pseudolabel_comparison_manifests",
]
