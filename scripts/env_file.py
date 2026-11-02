"""Configuration from a ``.env`` file, for the skill's optional integrations.

Two integrations of this skill are addressed by environment variables rather than
by command-line arguments: ``PDF2_DATA_DIR`` for the local ICDD PDF-2 card
database, and ``MP_API_KEY`` for the Materials Project. A variable is the right
place for such values because they belong to the machine, not to a single command
line -- but the variable still has to be set somewhere, and a shell session and an
agent session do not share one. A ``.env`` file found from the working directory
closes that gap.

This module is the single place in the skill where that file is read. Each
consumer calls :func:`load_env_file` immediately before reading its own variable,
so nothing depends on import order and no module mutates the process environment
merely by being imported.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from loguru import logger

__all__ = ["load_env_file"]


def load_env_file() -> Path | None:
    """Load ``.env`` from the working directory or one of its parents.

    A value already present in the process environment wins: ``load_dotenv`` is
    called without ``override``, so an explicitly exported variable is never
    replaced by a file entry. The ordering is deliberate -- the environment states
    what this particular run was told, the file states the machine's default.

    Calling it more than once is harmless.

    :returns: the file that was read, or ``None`` when there is none.
    """
    found = find_dotenv(usecwd=True)
    if not found:
        return None
    load_dotenv(found)
    logger.debug("Configuration file loaded: {}", found)
    return Path(found)
