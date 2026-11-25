"""Shared fixtures for the RSSidian test suite."""

import sys
from pathlib import Path

import pytest

# Ensure the repository root is importable when tests are run from testing/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def db_session(tmp_path):
    """Create an initialized SQLite database in a temporary directory."""
    from rssidian.models import init_db

    session = init_db(str(tmp_path / "rssidian-test.db"))
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def feed(db_session):
    """Create a single test feed."""
    from rssidian.models import Feed

    feed = Feed(title="Test Feed", url="https://example.com/feed.xml")
    db_session.add(feed)
    db_session.commit()
    return feed
