"""Keep every backend test on disposable storage before importing the app."""
import tempfile
import os
from pathlib import Path

import pytest
import config

# Unit tests never communicate with a real shared Windows bridge.
os.environ.pop("GRID_BOT_BRIDGE_DIR", None)
os.environ.pop("GRID_BOT_BRIDGE_TOKEN_FILE", None)

_storage = tempfile.TemporaryDirectory(prefix="grid-bot-tests-")
config.DB_PATH = str(Path(_storage.name) / "analyses.db")
config.RUNTIME_DIR = Path(_storage.name) / "runtime"
config.SETTINGS_PATH = config.RUNTIME_DIR / "settings.json"

import db


@pytest.fixture(scope="session", autouse=True)
def close_test_storage():
    yield
    db.engine.dispose()
    _storage.cleanup()


@pytest.fixture(autouse=True)
def clean_test_database():
    with db.engine.begin() as connection:
        connection.execute(db.Analysis.__table__.delete())
