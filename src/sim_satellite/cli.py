"""Command-line interface for Sim Satellite."""

import click

from sim_satellite import __version__
from sim_satellite.data.landcoverai import cli as landcoverai_cli
from sim_satellite.data.loveda import cli as loveda_cli
from sim_satellite.data.openearthmap import cli as openearthmap_cli
from sim_satellite.infer import cli as infer_cli
from sim_satellite.train import cli as train_cli
from sim_satellite.vae.training import cli as train_vae_cli
from sim_satellite.web import cli as web_cli


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="sim-satellite")
def cli() -> None:
    """Explore conditional satellite-image diffusion workflows."""


cli.add_command(landcoverai_cli)
cli.add_command(loveda_cli)
cli.add_command(openearthmap_cli)
cli.add_command(train_cli)
cli.add_command(train_vae_cli)
cli.add_command(infer_cli)
cli.add_command(web_cli)


if __name__ == "__main__":
    cli()
