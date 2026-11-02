"""Tests for the ``.env`` reader that configures the skill's optional integrations.

The integrations are addressed by environment variables (``PDF2_DATA_DIR``,
``MP_API_KEY``); the file exists so that a value can be declared once for a machine
instead of being exported in every shell. Two invariants matter and are checked
here: the file is actually read (the previous implementation imported ``dotenv``
inside a ``try`` that hid its absence, so the support looked present and did
nothing), and an exported variable outranks the file.
"""

from __future__ import annotations

import os
from pathlib import Path

from env_file import load_env_file

SETTING = "XRD_TEST_SETTING"


def test_env_file_is_read_from_the_working_directory(monkeypatch, tmp_path):
    """A declaration in ``.env`` must reach the process environment."""
    monkeypatch.delenv(SETTING, raising=False)
    (tmp_path / ".env").write_text(f"{SETTING}=from-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    loaded = load_env_file()

    assert loaded == tmp_path / ".env"
    assert os.environ[SETTING] == "from-file"


def test_exported_variable_wins_over_the_file(monkeypatch, tmp_path):
    """An explicitly exported variable must not be replaced by a file entry."""
    monkeypatch.setenv(SETTING, "from-environment")
    (tmp_path / ".env").write_text(f"{SETTING}=from-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    load_env_file()

    assert os.environ[SETTING] == "from-environment"


def test_absent_file_is_not_an_error(monkeypatch):
    """No ``.env`` anywhere is a normal state, not a failure.

    The search is stubbed rather than relying on the absence of a file in the
    temporary directory's ancestors, which the test cannot control.
    """
    monkeypatch.setattr("env_file.find_dotenv", lambda **_: "")
    assert load_env_file() is None


def test_path_is_reported_for_every_consumer(monkeypatch, tmp_path):
    """The loader returns the path, so a caller can log what configured it."""
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    loaded = load_env_file()

    assert isinstance(loaded, Path)
    assert loaded.is_file()
