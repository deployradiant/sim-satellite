"""Command-line interface for Sim Satellite."""

import click

from sim_satellite import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="sim-satellite")
def cli() -> None:
    """Explore conditional satellite-image diffusion workflows."""


@cli.command()
def data() -> None:
    """Prepare or inspect datasets."""
    click.echo("Data workflow placeholder.")


@cli.command()
def train() -> None:
    """Run model training."""
    click.echo("Training workflow placeholder.")


@cli.command()
def infer() -> None:
    """Run image generation or steering inference."""
    click.echo("Inference workflow placeholder.")


@cli.command()
def app() -> None:
    """Launch the drawing and generation app."""
    click.echo("Web app workflow placeholder.")


if __name__ == "__main__":
    cli()
