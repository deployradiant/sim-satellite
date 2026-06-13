"""Command-line interface for Sim Satellite."""

from typing import Any

import click

from sim_satellite import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="sim-satellite")
def cli() -> None:
    """Explore conditional satellite-image diffusion workflows."""


@cli.group()
def data() -> None:
    """Prepare or inspect datasets."""


def _add_optional_command(command: Any) -> None:
    if command is not None:
        cli.add_command(command)


from sim_satellite.data.openearthmap import cli as openearthmap_cli  # noqa: E402
from sim_satellite.infer import cli as infer_cli  # noqa: E402
from sim_satellite.train import cli as train_cli  # noqa: E402
from sim_satellite.web import cli as web_cli  # noqa: E402

data.add_command(openearthmap_cli)
_add_optional_command(train_cli)
_add_optional_command(infer_cli)
_add_optional_command(web_cli)


if __name__ == "__main__":
    cli()
