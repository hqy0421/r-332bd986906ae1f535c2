"""Tests for content provenance: which copy is adopted, why, and whether the
summary and embedding consume the same content version."""

from rssidian import provenance
from rssidian.models import Article, Feed

FULL_BODY = "Full article body. " * 80  # ~1520 chars after cleaning
SHORT_SUMMARY = "Short teaser summary. Read more"


class FakeConfig:
    """Minimal config surface used by the analysis code paths."""

    openrouter_api_key = "test-key"
    openrouter_prompt = "Summarize: {content}"
    openrouter_model = "test/model"
    openrouter_processing_model = "test/model"
    value_prompt_enabled = True
    value_prompt = "Rate: {content}"
    min_full_content_chars = 1000
    similarity_threshold = 0.65
    analyze_during_ingestion = False
    annoy_metric = "angular"
    annoy_n_trees = 2
    excerpt_length = 300
    annoy_index_path = "/tmp/rssidian-test-annoy.idx"

    def get(self, section, key, default=None):
        return getattr(self, key, default)


def make_entry(link="https://example.com/a", title="An Article", text=FULL_BODY, guid=None):
    return {
        "id": guid or link,
        "link": link,
        "title": title,
        "summary": text,
        "description": text,
        "published": "2026-11-20T10:00:00Z",
    }


def ingest(feed, db_session, entry, monkeypatch, fetch=None, config=None, analyze=False):
    """Run a single-entry ingestion with the network layer stubbed out."""
    from rssidian import rss

    monkeypatch.setattr(rss, "get_feed_entries", lambda *a, **k: [entry])
    calls = []

    def fake_fetch(url):
        calls.append(url)
        if fetch is None:
            return None
        return fetch(url)

    monkeypatch.setattr(rss, "fetch_jina_content", fake_fetch)
    result = rss.process_feed_result(
        feed,
        db_session,
        config=config,
        analyze_content=analyze,
    )
    return result, calls


def only_article(db_session):
    articles = db_session.query(Article).all()
    assert len(articles) == 1
    return articles[0]


def test_full_body_from_feed(feed, db_session, monkeypatch):
    result, calls = ingest(feed, db_session, make_entry(), monkeypatch)
    article = only_article(db_session)

    assert article.content_status == provenance.STATUS_FULL_TEXT
    assert article.content_source == provenance.CONTENT_SOURCE_FEED
    assert article.content_acceptance == provenance.ACCEPT_FEED_FULL
    assert article.content_hash == provenance.compute_content_hash(article.content)
    assert article.jina_enhanced is False
    assert calls == []  # no supplemental fetch for a full body
    assert result["new_articles"] == 1


def test_summary_only_when_fetch_fails(feed, db_session, monkeypatch):
    result, calls = ingest(
        feed, db_session, make_entry(text=SHORT_SUMMARY), monkeypatch, fetch=None
    )
    article = only_article(db_session)

    assert calls == ["https://example.com/a"]
    assert article.content_status == provenance.STATUS_SUMMARY_ONLY
    assert article.content_source == provenance.CONTENT_SOURCE_SUMMARY
    assert article.content_acceptance == provenance.ACCEPT_FETCH_FAILED_KEPT_SUMMARY
    assert article.source_public_url == "https://example.com/a"
    assert "Short teaser summary" in article.content
    assert article.jina_enhanced is False


def test_summary_upgraded_by_supplemental_fetch(feed, db_session, monkeypatch):
    result, calls = ingest(
        feed,
        db_session,
        make_entry(text=SHORT_SUMMARY),
        monkeypatch,
        fetch=lambda url: FULL_BODY,
    )
    article = only_article(db_session)

    assert article.content_status == provenance.STATUS_FULL_TEXT
    assert article.content_source == provenance.CONTENT_SOURCE_SUPPLEMENTAL
    assert article.content_acceptance == provenance.ACCEPT_SUPPLEMENTAL_LONGER
    assert article.jina_enhanced is True
    assert article.content_hash == provenance.compute_content_hash(article.content)


def test_cross_subscription_upgrade(feed, db_session, monkeypatch):
    # Subscription A: summary only, supplemental fetch fails -> summary_only row
    ingest(feed, db_session, make_entry(text=SHORT_SUMMARY), monkeypatch, fetch=None)
    original = only_article(db_session)
    assert original.content_status == provenance.STATUS_SUMMARY_ONLY
    assert original.content_version == 1
    original_id = original.id

    # Simulate having derived a summary/quality/embedding from the old copy.
    original.summary = "old summary"
    original.summary_content_hash = original.content_hash
    original.quality_tier = "C"
    original.quality_score = 30
    original.embedding_generated = True
    original.embedding_vector = "[0.1, 0.2]"
    original.embedding_content_hash = original.content_hash
    original.processed = True
    db_session.commit()

    # Subscription B: same URL, now carries the full body.
    other_feed = Feed(title="Other Feed", url="https://example.org/feed.xml")
    db_session.add(other_feed)
    db_session.commit()
    result, _ = ingest(
        other_feed,
        db_session,
        make_entry(text=FULL_BODY, guid="other-guid"),
        monkeypatch,
        fetch=None,
    )

    db_session.expire_all()
    assert db_session.query(Article).count() == 1
    upgraded = db_session.query(Article).one()
    assert upgraded.id == original_id
    assert upgraded.content_status == provenance.STATUS_FULL_TEXT
    assert upgraded.content_acceptance == provenance.ACCEPT_UPGRADED
    assert upgraded.content_version == 2
    assert upgraded.content_source_feed_id == other_feed.id
    assert upgraded.content_hash == provenance.compute_content_hash(upgraded.content)
    assert "Full article body." in upgraded.content
    assert len(upgraded.content) >= 1000
    # Derived artifacts must be invalidated so they regenerate from the new copy.
    assert upgraded.summary is None
    assert upgraded.summary_content_hash is None
    assert upgraded.quality_tier is None
    assert upgraded.embedding_generated is False
    assert upgraded.embedding_vector is None
    assert upgraded.embedding_content_hash is None
    assert result["upgraded_articles"] == 1


def test_summary_copy_does_not_trigger_upgrade_loop(feed, db_session, monkeypatch):
    _, first_calls = ingest(
        feed, db_session, make_entry(text=SHORT_SUMMARY), monkeypatch, fetch=None
    )
    article = only_article(db_session)
    first_version = article.content_version

    # Run the same feed again; still summary-only, no upgrade, no duplicate.
    _, second_calls = ingest(
        feed, db_session, make_entry(text=SHORT_SUMMARY), monkeypatch, fetch=None
    )
    db_session.expire_all()
    assert db_session.query(Article).count() == 1
    assert db_session.query(Article).one().content_version == first_version
    # The supplemental fetch is attempted once, not on every ingest.
    assert len(first_calls) == 1
    assert second_calls == []


def test_legacy_row_is_classified_without_version_bump(feed, db_session, monkeypatch):
    """A pre-provenance row gets source/status recorded when re-ingested."""
    legacy = Article(
        feed_id=feed.id,
        title="An Article",
        url="https://example.com/a",
        guid="guid-a",
        content=SHORT_SUMMARY,  # raw (uncleaned) legacy text
        processed=True,
    )
    db_session.add(legacy)
    db_session.commit()

    result, calls = ingest(
        feed, db_session, make_entry(text=SHORT_SUMMARY, guid="guid-a"), monkeypatch
    )
    db_session.expire_all()
    stored = db_session.query(Article).one()

    assert result["upgraded_articles"] == 0
    assert stored.content_status == provenance.STATUS_SUMMARY_ONLY
    assert stored.content_source == provenance.CONTENT_SOURCE_SUMMARY
    assert stored.content_source_feed_id == feed.id
    assert stored.content_version == 1  # unchanged text, no version bump
    assert stored.content_hash == provenance.compute_content_hash(stored.content)

    # A second ingest must not re-fetch from the same feed.
    _, calls_again = ingest(
        feed, db_session, make_entry(text=SHORT_SUMMARY, guid="guid-a"), monkeypatch
    )
    assert calls_again == []


def test_same_run_duplicate_url_is_deduplicated(feed, db_session, monkeypatch):
    from rssidian import rss

    entry_a = make_entry(guid="guid-a")
    entry_b = make_entry(guid="guid-b")  # different guid, same link
    monkeypatch.setattr(rss, "get_feed_entries", lambda *a, **k: [entry_a, entry_b])
    monkeypatch.setattr(rss, "fetch_jina_content", lambda url: None)

    rss.process_feed_result(feed, db_session, config=None, analyze_content=False)

    assert db_session.query(Article).count() == 1


def test_summary_and_embedding_share_content_version(feed, db_session, monkeypatch):
    from rssidian.core import RSSProcessor

    article = Article(
        feed_id=feed.id,
        title="An Article",
        url="https://example.com/a",
        guid="guid-a",
        description="desc",
        content=FULL_BODY,
        content_hash=provenance.compute_content_hash(FULL_BODY),
        content_source=provenance.CONTENT_SOURCE_FEED,
        content_status=provenance.STATUS_FULL_TEXT,
        content_version=1,
    )
    db_session.add(article)
    db_session.commit()

    processor = RSSProcessor(FakeConfig(), db_session)

    captured = {"summary_inputs": [], "embedding_inputs": []}
    monkeypatch.setattr(
        processor,
        "_generate_embedding",
        lambda text: captured["embedding_inputs"].append(text) or [0.0] * 768,
    )

    def fake_api(prompt, model=None):
        captured["summary_inputs"].append(prompt)
        return "A summary."

    monkeypatch.setattr(processor, "_call_openrouter_api", fake_api)
    # Annoy rebuild is irrelevant to this assertion.
    monkeypatch.setattr(processor, "_flush_embeddings", lambda: 0)

    processor.process_articles()

    db_session.expire_all()
    stored = db_session.query(Article).one()
    expected = provenance.select_processing_text(stored)
    assert stored.summary_content_hash == stored.content_hash
    assert stored.embedding_content_hash == stored.content_hash
    assert all(expected in prompt for prompt in captured["summary_inputs"])
    assert captured["embedding_inputs"] and captured["embedding_inputs"][0] == expected


def test_stale_summary_is_refreshed_from_new_content_version(
    feed, db_session, monkeypatch
):
    """A summary stamped with an older content hash is regenerated in place."""
    from rssidian.core import RSSProcessor

    article = Article(
        feed_id=feed.id,
        title="An Article",
        url="https://example.com/a",
        guid="guid-a",
        content=FULL_BODY,
        content_hash=provenance.compute_content_hash(FULL_BODY),
        content_status=provenance.STATUS_FULL_TEXT,
        content_version=2,
        summary="summary of the old copy",
        summary_content_hash="stale-hash",
        processed=True,
    )
    db_session.add(article)
    db_session.commit()

    processor = RSSProcessor(FakeConfig(), db_session)
    seen = []

    def fake_summary(art):
        seen.append(provenance.select_processing_text(art))
        return "summary of the new copy"

    monkeypatch.setattr(processor, "_generate_summary", fake_summary)
    monkeypatch.setattr(processor, "_generate_embedding", lambda text: [0.0] * 768)
    monkeypatch.setattr(processor, "_flush_embeddings", lambda: 0)
    monkeypatch.setattr(processor, "_call_openrouter_api", lambda p, model=None: "x")

    processor.process_articles()

    db_session.expire_all()
    stored = db_session.query(Article).one()
    assert seen == [FULL_BODY]
    assert stored.summary == "summary of the new copy"
    assert stored.summary_content_hash == stored.content_hash

    # A second run finds nothing stale and makes no further API call.
    processor.process_articles()
    assert len(seen) == 1


def test_empty_content_hash_is_none():
    assert provenance.compute_content_hash("") is None
    assert provenance.compute_content_hash("   ") is None
    assert provenance.compute_content_hash("text") == provenance.compute_content_hash("text")


def test_select_processing_text_fallbacks():
    class Row:
        def __init__(self, content=None, description=None, summary=None):
            self.content, self.description, self.summary = content, description, summary

    assert provenance.select_processing_text(Row(content="c", description="d")) == "c"
    assert provenance.select_processing_text(Row(description="d", summary="s")) == "d"
    assert provenance.select_processing_text(Row(summary="s")) == "s"
    assert provenance.select_processing_text(Row()) is None
