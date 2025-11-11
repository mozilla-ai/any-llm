import logging
import os
import re
import shutil
import subprocess
import sys

import click
import uvicorn
from uvicorn.config import logger

from any_llm.gateway.config import load_config
from any_llm.gateway.log_config import setup_logger
from any_llm.gateway.server import create_app


@click.group()
def cli() -> None:
    """any-llm-gateway CLI."""


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config YAML file",
    default=None,
)
@click.option("--host", default=None, help="Host to bind the server to")
@click.option("--port", default=None, type=int, help="Port to bind the server to")
@click.option("--database-url", envvar="DATABASE_URL", help="Database connection URL")
@click.option("--master-key", envvar="GATEWAY_MASTER_KEY", help="Master key for management endpoints")
@click.option("--auto-migrate/--no-auto-migrate", default=None, help="Automatically run database migrations on startup")
@click.option("--workers", default=1, type=int, help="Number of worker processes")
@click.option(
    "--log-level",
    default=logging.INFO,
    type=click.Choice([logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]),
    help="Logging level",
)
def serve(
    config: str | None,
    host: str | None,
    port: int | None,
    database_url: str | None,
    master_key: str | None,
    auto_migrate: bool | None,
    workers: int,
    log_level: int,
) -> None:
    """Start the gateway server."""
    gateway_config = load_config(config)
    setup_logger(level=log_level)

    if host:
        gateway_config.host = host
    if port:
        gateway_config.port = port
    if database_url:
        gateway_config.database_url = database_url
    if master_key:
        gateway_config.master_key = master_key
    if auto_migrate is not None:
        gateway_config.auto_migrate = auto_migrate

    if not gateway_config.master_key:
        logger.warning(
            "No master key configured. Key management endpoints will be unavailable.",
        )
        logger.warning("Set GATEWAY_MASTER_KEY environment variable or use --master-key flag.")

    logger.info(f"Starting any-llm-gateway on {gateway_config.host}:{gateway_config.port}")
    logger.info(f"Database: {gateway_config.database_url}")

    if gateway_config.providers:
        logger.info(f"Configured providers: {', '.join(gateway_config.providers.keys())}")

    app = create_app(gateway_config)

    try:
        uvicorn.run(
            app,
            host=gateway_config.host,
            port=gateway_config.port,
            workers=workers,
        )
    except KeyboardInterrupt:
        logger.info("\nShutting down gateway...")
        sys.exit(0)


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--database-url", envvar="DATABASE_URL", help="Database connection URL")
def init_db(config: str | None, database_url: str | None) -> None:
    """Initialize the database schema."""
    from any_llm.gateway.db import init_db as db_init

    gateway_config = load_config(config)

    if database_url:
        gateway_config.database_url = database_url

    click.echo(f"Initializing database: {gateway_config.database_url}")

    db_init(gateway_config.database_url)

    click.echo("Database initialized successfully!")


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to config YAML file")
@click.option("--database-url", envvar="DATABASE_URL", help="Database connection URL")
@click.option("--revision", default="head", help="Target revision (default: head)")
def migrate(config: str | None, database_url: str | None, revision: str) -> None:
    """Run database migrations using Alembic."""
    gateway_config = load_config(config)

    if database_url:
        gateway_config.database_url = database_url

    if not re.match(r"^[a-zA-Z0-9_+\-]+$", revision):
        click.echo(f"Invalid revision format: {revision}", err=True)
        sys.exit(1)

    alembic_path = shutil.which("alembic")
    if not alembic_path:
        click.echo("alembic command not found in PATH", err=True)
        sys.exit(1)

    click.echo(f"Running migrations on: {gateway_config.database_url}")
    click.echo(f"Target revision: {revision}")

    env = os.environ.copy()
    env["GATEWAY_DATABASE_URL"] = gateway_config.database_url

    try:
        result = subprocess.run(  # noqa: S603 validated up a few lines
            [alembic_path, "upgrade", revision],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        click.echo(result.stdout)
        click.echo("Migrations completed successfully!")
    except subprocess.CalledProcessError as e:
        click.echo(f"Migration failed: {e.stderr}", err=True)
        sys.exit(1)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
