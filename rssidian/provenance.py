"""Content provenance: decide which copy of an article is adopted, why, and
which version the summary/embedding were derived from.

This module is deliberately self-contained (it imports nothing from ``rss`` or
``core``) so that both the ingestion path and the analysis path can depend on
it without creating import cycles. Callers inject the HTML cleaner and the
supplemental fetch function.
"""

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# Content sources
CONTENT_SOURCE_FEED = "feed_content"
CONTENT_SOURCE_SUMMARY = "feed_summary"
CONTENT_SOURCE_SUPPLEMENTAL = "supplemental_fetch"
CONTENT_SOURCE_LEGACY = "legacy_unknown"

# Content status (distinguishes full body from summary-only)
STATUS_FULL_TEXT = "full_text"
STATUS_SUMMARY_ONLY = "summary_only"
STATUS_UNKNOWN = "unknown"

# Acceptance basis codes
ACCEPT_FEED_FULL = "feed_meets_threshold"
ACCEPT_SUPPLEMENTAL_LONGER = "supplemental_longer"
ACCEPT_FETCH_FAILED_KEPT_SUMMARY = "supplemental_fetch_failed_retained_summary"
ACCEPT_FETCH_NOT_LONGER_KEPT_SUMMARY = "supplemental_not_longer_retained_summary"
ACCEPT_FETCH_SKIPPED = "fetch_skipped_no_url"
ACCEPT_NO_CONTENT = "no_content_available"
ACCEPT_UPGRADED = "upgraded_from_duplicate"

DEFAULT_FULL_CONTENT_THRESHOLD = 1000
DEFAULT_TRUNCATION_MARKERS = ("[...]", "read more", "continue reading")


def compute_content_hash(text: Optional[str]) -> Optional[str]:
    """Return the SHA-256 of the exact text that will be stored.

    Empty/whitespace-only text hashes to ``None`` (never ``sha256("")``) so
    that "no content version" is distinguishable from "empty version".
    """
    if not text or not text.strip():
        return None
    normalized = unicodedata.normalize("NFC", text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def select_processing_text(article: Any) -> Optional[str]:
    """Return the single text summary/embedding/search must operate on.

    Prefers the stored full content, falling back to the feed description and
    then the summary so that summary-only articles still get an embedding.
    """
    for attr in ("content", "description", "summary"):
        value = getattr(article, attr, None)
        if value and value.strip():
            return value
    return None


def extract_entry_text(entry: Dict[str, Any]) -> str:
    """Extract the main text from a feed entry using the historical field order."""
    content = ""

    if entry.get("content"):
        for content_item in entry["content"]:
            value = content_item.get("value") if isinstance(content_item, dict) else None
            if value:
                content = value
                if len(content) > 500:
                    return content

    if not content and entry.get("summary_detail"):
        content = entry["summary_detail"].get("value", "") or ""

    if not content and entry.get("summary"):
        content = entry["summary"]

    if not content and entry.get("description"):
        content = entry["description"]

    if not content and entry.get("content_encoded"):
        content = entry["content_encoded"]

    if not content and entry.get("value"):
        content = entry["value"]

    return content


def assess_content_text(
    text: str,
    threshold: int = DEFAULT_FULL_CONTENT_THRESHOLD,
    markers: Tuple[str, ...] = DEFAULT_TRUNCATION_MARKERS,
) -> Tuple[bool, List[str]]:
    """Classify text as full body vs. summary.

    Returns ``(is_full, matched_markers)``. Text is considered full only when
    it is long enough *and* contains no truncation markers.
    """
    if not text or not text.strip():
        return False, []

    lowered = text.lower()
    matched = [marker for marker in markers if marker.lower() in lowered]
    is_full = len(text) >= threshold and not matched
    return is_full, matched


@dataclass
class ContentAssessment:
    """The outcome of deciding which content copy to adopt."""

    text: str
    source: str
    status: str
    acceptance: str
    source_url: Optional[str] = None
    fetch_attempted: bool = False
    fetch_succeeded: bool = False
    feed_chars: int = 0
    fetched_chars: int = 0
    matched_markers: List[str] = field(default_factory=list)
    threshold: int = DEFAULT_FULL_CONTENT_THRESHOLD
    content_hash: Optional[str] = None

    def detail_json(self) -> str:
        """Serialize the audit details for ``Article.content_acceptance_detail``."""
        return json.dumps(
            {
                "feed_chars": self.feed_chars,
                "fetched_chars": self.fetched_chars,
                "threshold": self.threshold,
                "matched_markers": self.matched_markers,
                "fetch_attempted": self.fetch_attempted,
                "fetch_succeeded": self.fetch_succeeded,
                "status": self.status,
            },
            ensure_ascii=False,
        )


def adopt_entry_content(
    entry: Dict[str, Any],
    *,
    source_url: Optional[str],
    fetch_supplemental: Callable[[str], Optional[str]],
    clean_fn: Callable[[str], str],
    threshold: int = DEFAULT_FULL_CONTENT_THRESHOLD,
    fetch_if_not_full: bool = True,
) -> ContentAssessment:
    """Pick the content copy for a feed entry and record why it was chosen.

    The feed copy is cleaned once and classified. If it is only a summary and a
    supplemental fetch is possible, the fetched copy is cleaned and adopted
    only when it is strictly longer (preserving the previous behavior). When the
    fetch fails or returns nothing better, the feed copy is retained and the
    acceptance basis makes that explicit.
    """
    raw_feed_text = extract_entry_text(entry)
    feed_text = clean_fn(raw_feed_text) if raw_feed_text else ""
    feed_is_full, feed_markers = assess_content_text(feed_text, threshold)

    assessment = ContentAssessment(
        text=feed_text,
        source=CONTENT_SOURCE_FEED,
        status=STATUS_FULL_TEXT if feed_is_full else STATUS_SUMMARY_ONLY,
        acceptance=ACCEPT_FEED_FULL,
        source_url=source_url,
        feed_chars=len(feed_text),
        matched_markers=list(feed_markers),
        threshold=threshold,
    )

    if not feed_text:
        assessment.source = CONTENT_SOURCE_FEED
        assessment.status = STATUS_UNKNOWN
        assessment.acceptance = ACCEPT_NO_CONTENT

    if feed_is_full or not fetch_if_not_full:
        assessment.content_hash = compute_content_hash(assessment.text)
        return assessment

    if not source_url or not source_url.startswith("http"):
        assessment.acceptance = ACCEPT_FETCH_SKIPPED
        assessment.content_hash = compute_content_hash(assessment.text)
        return assessment

    assessment.fetch_attempted = True
    try:
        fetched_raw = fetch_supplemental(source_url)
    except Exception:
        fetched_raw = None

    fetched_text = clean_fn(fetched_raw) if fetched_raw else ""
    assessment.fetched_chars = len(fetched_text)

    if fetched_text and len(fetched_text) > len(feed_text):
        assessment.fetch_succeeded = True
        assessment.text = fetched_text
        assessment.source = CONTENT_SOURCE_SUPPLEMENTAL
        fetched_is_full, fetched_markers = assess_content_text(fetched_text, threshold)
        assessment.status = STATUS_FULL_TEXT if fetched_is_full else STATUS_SUMMARY_ONLY
        assessment.matched_markers = list(fetched_markers)
        assessment.acceptance = ACCEPT_SUPPLEMENTAL_LONGER
    elif assessment.text:
        assessment.source = CONTENT_SOURCE_SUMMARY
        assessment.status = STATUS_SUMMARY_ONLY
        assessment.acceptance = (
            ACCEPT_FETCH_NOT_LONGER_KEPT_SUMMARY
            if fetched_text
            else ACCEPT_FETCH_FAILED_KEPT_SUMMARY
        )
    else:
        # Nothing usable from the feed or the supplemental fetch.
        assessment.source = CONTENT_SOURCE_FEED
        assessment.status = STATUS_UNKNOWN
        assessment.acceptance = ACCEPT_NO_CONTENT

    assessment.content_hash = compute_content_hash(assessment.text)
    return assessment


def should_upgrade(existing: Any, incoming: ContentAssessment) -> bool:
    """Whether an incoming full copy should replace summary-only stored content."""
    if incoming.status != STATUS_FULL_TEXT:
        return False
    if not incoming.text:
        return False
    return getattr(existing, "content_status", None) != STATUS_FULL_TEXT


def apply_content_adoption(
    article: Any,
    assessment: ContentAssessment,
    *,
    source_feed_id: Optional[int] = None,
    is_new: bool = False,
    acceptance: Optional[str] = None,
) -> bool:
    """Write an adoption decision onto an Article.

    For an existing row whose stored text actually changes, bumps
    ``content_version`` and invalidates summary/value/embedding artifacts so
    they are regenerated from the new version. Returns whether content changed.
    """
    new_text = assessment.text or ""
    old_text = article.content or ""
    changed = is_new or old_text != new_text

    article.content = new_text
    article.content_source = assessment.source
    article.content_status = assessment.status
    article.content_acceptance = acceptance or assessment.acceptance
    article.content_acceptance_detail = assessment.detail_json()
    article.content_hash = assessment.content_hash or compute_content_hash(new_text)
    article.source_public_url = assessment.source_url or getattr(article, "source_public_url", None)
    if source_feed_id is not None:
        article.content_source_feed_id = source_feed_id
    article.word_count = len(new_text.split()) if new_text else 0
    article.jina_enhanced = assessment.source == CONTENT_SOURCE_SUPPLEMENTAL

    if is_new:
        article.content_version = 1
        return True

    if changed:
        article.content_version = (getattr(article, "content_version", None) or 1) + 1
        article.summary = None
        article.summary_content_hash = None
        article.quality_tier = None
        article.quality_score = None
        article.labels = None
        article.processed = False
        article.processed_at = None
        article.embedding_generated = False
        article.embedding_vector = None
        article.embedding_content_hash = None
        return True

    return False
