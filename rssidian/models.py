from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session

Base = declarative_base()


class Feed(Base):
    """Model for RSS feed subscriptions."""
    __tablename__ = "feeds"

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    url = Column(String(512), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    website_url = Column(String(512), nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    last_checked = Column(DateTime, default=datetime.utcnow)
    muted = Column(Boolean, default=False)
    muted_reason = Column(Text, nullable=True)  # Reason for muting the feed
    peer_through = Column(Boolean, default=False)  # Whether to peer through aggregator to origin article
    error_count = Column(Integer, default=0)  # Count of consecutive errors
    no_entries_count = Column(Integer, default=0)  # Count of consecutive checks with no entries
    last_error = Column(Text, nullable=True)
    
    # Statistics
    article_count = Column(Integer, default=0)  # Total count of ingested articles
    avg_quality_score = Column(Float, nullable=True)  # Average quality score of articles
    quality_tier_counts = Column(Text, nullable=True)  # JSON string of quality tier counts {"S": 5, "A": 10, ...}
    
    # Relationships
    articles = relationship("Article", back_populates="feed", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Feed(id={self.id}, title='{self.title}', muted={self.muted}, peer_through={self.peer_through})>"


class Article(Base):
    """Model for articles from RSS feeds."""
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True)
    feed_id = Column(Integer, ForeignKey("feeds.id"), nullable=False)
    title = Column(String(512), nullable=False)
    url = Column(String(512), nullable=False)
    guid = Column(String(512), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    author = Column(String(255), nullable=True)
    published_at = Column(DateTime, nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime, nullable=True)
    summary = Column(Text, nullable=True)
    quality_tier = Column(String(1), nullable=True)  # S, A, B, C, D
    quality_score = Column(Integer, nullable=True)  # 1-100
    labels = Column(String(255), nullable=True)
    embedding_generated = Column(Boolean, default=False)
    embedding_vector = Column(Text, nullable=True)  # JSON-serialized embedding vector
    word_count = Column(Integer, nullable=True)
    jina_enhanced = Column(Boolean, default=False)  # Flag to indicate if content was fetched from Jina.ai

    # Content provenance: which copy was adopted, why, and which version
    # the summary/embedding were derived from.
    content_source = Column(String(32), nullable=True)  # feed_content | feed_summary | supplemental_fetch | legacy_unknown
    content_status = Column(
        String(16), nullable=False, server_default=text("'unknown'")
    )  # full_text | summary_only | unknown
    content_acceptance = Column(String(64), nullable=True)  # acceptance reason code
    content_acceptance_detail = Column(Text, nullable=True)  # JSON audit blob
    content_hash = Column(String(64), nullable=True)  # sha256 of the stored content
    content_version = Column(Integer, nullable=False, server_default=text("1"))
    source_public_url = Column(String(512), nullable=True, index=True)  # public original URL
    content_source_feed_id = Column(Integer, nullable=True)  # feed that supplied the adopted copy
    summary_content_hash = Column(String(64), nullable=True)  # content version the summary came from
    embedding_content_hash = Column(String(64), nullable=True)  # content version the embedding came from

    # Relationships
    feed = relationship("Feed", back_populates="articles")

    def __repr__(self) -> str:
        return f"<Article(id={self.id}, title='{self.title[:30]}...', quality_tier='{self.quality_tier or 'None'}')>"


# Columns added after the initial schema. Each entry is (name, DDL type).
_ARTICLE_MIGRATION_COLUMNS = [
    ("embedding_vector", "TEXT"),
    ("content_source", "VARCHAR(32)"),
    ("content_status", "VARCHAR(16) NOT NULL DEFAULT 'unknown'"),
    ("content_acceptance", "VARCHAR(64)"),
    ("content_acceptance_detail", "TEXT"),
    ("content_hash", "VARCHAR(64)"),
    ("content_version", "INTEGER NOT NULL DEFAULT 1"),
    ("source_public_url", "VARCHAR(512)"),
    ("content_source_feed_id", "INTEGER"),
    ("summary_content_hash", "VARCHAR(64)"),
    ("embedding_content_hash", "VARCHAR(64)"),
]


def _migrate_article_columns(engine) -> None:
    """Add any missing columns to an existing articles table."""
    from sqlalchemy import inspect

    inspector = inspect(engine)
    if "articles" not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns("articles")}
    with engine.connect() as conn:
        for name, ddl_type in _ARTICLE_MIGRATION_COLUMNS:
            if name not in existing:
                conn.execute(text(f"ALTER TABLE articles ADD COLUMN {name} {ddl_type}"))
        # create_all() creates this index, but ALTER TABLE above does not.
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_articles_source_public_url "
            "ON articles (source_public_url)"
        ))
        conn.commit()


def init_db(db_path: str) -> Session:
    """Initialize database and return session."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    _migrate_article_columns(engine)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def get_db_session(db_path: str) -> Session:
    """Get a database session."""
    engine = create_engine(f"sqlite:///{db_path}")

    _migrate_article_columns(engine)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()