"""Command-line interface for Sim Satellite."""

import click

from sim_satellite import __version__
from sim_satellite.data.openearthmap import cli as openearthmap_cli
from sim_satellite.data.sam_training import cli as sam_data_cli
from sim_satellite.infer import cli as infer_cli
from sim_satellite.sam_classifier import label_cli as sam_classifier_label_cli
from sim_satellite.sam_classifier import predict_cli as sam_classifier_predict_cli
from sim_satellite.sam_classifier import train_cli as sam_classifier_train_cli
from sim_satellite.sam_compare import compare_cli as sam_compare_cli
from sim_satellite.sam_compare import compare_models_cli as sam_compare_models_cli
from sim_satellite.sam_compare import evaluate_cli as sam_evaluate_cli
from sim_satellite.sam_compare import propose_cli as sam_propose_cli
from sim_satellite.sam_finetune import cli as sam_finetune_cli
from sim_satellite.segmentation_compare import cli as segmentation_compare_cli
from sim_satellite.segmentation_train import cli as segmentation_train_cli
from sim_satellite.train import cli as train_cli
from sim_satellite.vae_compare import cli as vae_compare_cli
from sim_satellite.web import cli as web_cli


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="sim-satellite")
def cli() -> None:
    """Explore conditional satellite-image diffusion workflows."""


cli.add_command(openearthmap_cli)
cli.add_command(sam_data_cli)
cli.add_command(sam_propose_cli)
cli.add_command(sam_evaluate_cli)
cli.add_command(sam_compare_cli)
cli.add_command(sam_compare_models_cli)
cli.add_command(sam_finetune_cli)
cli.add_command(sam_classifier_label_cli)
cli.add_command(sam_classifier_train_cli)
cli.add_command(sam_classifier_predict_cli)
cli.add_command(segmentation_compare_cli)
cli.add_command(segmentation_train_cli)
cli.add_command(train_cli)
cli.add_command(infer_cli)
cli.add_command(vae_compare_cli)
cli.add_command(web_cli)


if __name__ == "__main__":
    cli()
