"""Core pipeline logic for the Customer Sentiment Signal project."""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import yaml

try:
    import spacy
except ImportError:  # pragma: no cover - tests will mock
    spacy = None

try:
    from transformers import pipeline as hf_pipeline
except ImportError:  # pragma: no cover - tests will mock
    hf_pipeline = None

LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = [
    "id",
    "source",
    "customer",
    "date",
    "agent",
    "service_product",
    "program",
    "incident_id",
    "activity",
    "notes",
]


@dataclass
class Signal:
    signal_id: str
    signal: str
    weight: float
    signal_score: float
    signal_sentiment: str


_SENTINEL = object()
_CONJUNCTION_SPLITTER = re.compile(r"\b(?:but|however|although|though|yet|even though)\b", re.IGNORECASE)
_GREETING_PREFIXES = ("hi", "hello", "thanks", "thank you", "dear", "greetings")
_GREETING_TOKEN_LIMIT = 12
_GREETING_ONLY_TERMS = {
    "hi",
    "hello",
    "thanks",
    "thank",
    "thankyou",
    "dear",
    "greetings",
    "for",
    "attending",
    "the",
    "call",
    "meeting",
    "team",
    "time",
    "following",
    "your",
    "support",
    "appreciate",
    "always",
    "as",
    "quick",
    "fix",
    "hopefully",
}
# Matches social/courtesy phrases in original text (before fluff removal).
# Used to catch closings like "Thanks for your support" and vague fillers like "Hopefully a quick fix"
# that lose their greeting word after fluff cleaning and would otherwise slip through.
_SOCIAL_PATTERNS = re.compile(
    r"^\s*(?:"
    r"thanks?\s+for\b"
    r"|thank\s+you\s+for\b"
    r"|hopefully\b"
    r"|(?:just\s+a?\s*)?heads?\s*up\b"
    r"|appreciate\s+(?:your|the)\b"
    r"|looking\s+forward\b"
    r")",
    re.IGNORECASE,
)
_MIN_SIGNAL_TOKENS = 3
# Detects technical log/error content that carries no sentiment value:
# error log fields, ODBC/SQL Server tags, SQL return codes, line/column references.
_TECHNICAL_LOG_PATTERNS = re.compile(
    r"(?:retcode\s*:|sqlstate\s*:|nativeerror\s*:|"
    r"\[microsoft\]|\[odbc|\[sql\s+server\]|"
    r"\bsql_error\b|\bsql_success\b|"
    r"line\s*:\s*-?\d+\s+column\s*:\s*-?\d+|"
    r"\bcannot\s+insert\b)",          # SQL error message bodies
    re.IGNORECASE,
)
# Strips leading/trailing numbered list markers: "3. Signal text" → "Signal text",
# "Signal text 3" → "Signal text"
_LIST_MARKER = re.compile(r"(?:^\s*\d+[\.\)]\s*|\s+\d+\s*$)")
# Sentences matching these patterns are pure background narrative — timeline facts,
# reproduction steps, or test descriptions — that carry no customer judgment.
# The whitelist approach (requiring impact phrases) was too narrow and silently
# dropped everyday feedback like "too slow", "failing", "Fantastic support".
# A blacklist is safer: filter only what we can positively identify as noise.
_BACKGROUND_PATTERNS = re.compile(
    r"(?:"
    r"\bwe\s+(?:also\s+)?tested\b"
    r"|\bwe\s+were\s+able\s+to\s+reproduce\b"
    r"|\bit\s+was\s+only\s+through\b"
    r"|\bthis\s+is\s+how\b"
    r"|\bwere\s+subsequently\b"
    r"|\bwere\s+(?:added|changed|loaded|released|updated|modified)\s+(?:to|during|in|on)\b"
    r")",
    re.IGNORECASE,
)
# Stopwords excluded from Jaccard similarity during deduplication.
_DEDUP_STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
    "is", "was", "were", "be", "been", "are", "it", "its", "this", "that",
    "these", "those", "with", "by", "from", "as", "into", "through", "not",
    "we", "our", "they", "their", "has", "have", "had", "also", "only",
    "even", "but", "which", "who", "what", "how", "all", "due",
}


def _deduplicate_signals(texts: List[str]) -> List[str]:
    """Remove near-duplicate signals using Jaccard similarity on meaningful words.

    Two signals are considered duplicates when they share more than 60 % of
    their non-stopword vocabulary.  The first-encountered signal is kept so
    that the numbered-list items (which appear earlier) take precedence over
    the prose restatements that follow.
    """
    def meaningful(text: str) -> set:
        return {w for w in text.lower().split() if w not in _DEDUP_STOPWORDS and len(w) > 2}

    kept: List[str] = []
    for text in texts:
        words = meaningful(text)
        duplicate = any(
            len(words & meaningful(k)) / len(words | meaningful(k)) >= 0.6
            for k in kept
            if words and meaningful(k)
        )
        if not duplicate:
            kept.append(text)
    return kept


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def validate_schema(frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_spacy_model() -> "spacy.language.Language":
    if spacy is None:
        raise RuntimeError("spaCy is not installed; install it to enable token extraction")
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return spacy.blank("en")


def build_sentiment_pipeline():
    if hf_pipeline is None:
        raise RuntimeError("transformers is not installed; install it to enable sentiment scoring")
    return hf_pipeline("sentiment-analysis")


def clean_text(text: str, fluff_terms: Iterable[str]) -> str:
    if not text:
        return ""
    normalized = text.lower()
    for term in fluff_terms:
        # Use word boundaries to avoid corrupting words that contain fluff terms
        # e.g. "hi" must not remove "hi" from "this" → "t s"
        normalized = re.sub(r"\b" + re.escape(term) + r"\b", " ", normalized)
    normalized = re.sub(r"[^\w\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _split_on_conjunctions(text: str) -> List[str]:
    parts = [piece.strip() for piece in _CONJUNCTION_SPLITTER.split(text)]
    return [part for part in parts if part]


def chunk_notes(text: str, nlp_model) -> List[str]:
    if not text:
        return []
    if nlp_model is None:
        return _split_on_conjunctions(text)
    doc = nlp_model(text)
    sentences = list(doc.sents) or [doc]
    chunks: List[str] = []
    for sent in sentences:
        chunks.extend(_split_on_conjunctions(sent.text))
    return [chunk for chunk in chunks if chunk]


def _is_greeting_chunk(text: str, fluff_terms: Iterable[str]) -> bool:
    # Check original text first — social patterns like "Thanks for your support"
    # lose their greeting word after fluff cleaning, so we must catch them here.
    if _SOCIAL_PATTERNS.match(text.strip()):
        return True
    cleaned = clean_text(text, fluff_terms)
    if not cleaned:
        return True
    tokens = cleaned.split()
    if len(tokens) > _GREETING_TOKEN_LIMIT:
        return False
    lower = cleaned.lower().split()
    if lower[0] not in _GREETING_PREFIXES:
        return False
    cleaned_tokens = [token.strip(".,!") for token in lower]
    if all(token in _GREETING_ONLY_TERMS for token in cleaned_tokens):
        return True
    return False


def _filter_signal_candidates(chunks: Iterable[str], fluff_terms: Iterable[str]) -> List[str]:
    filtered: List[str] = []
    for chunk in chunks:
        if _is_greeting_chunk(chunk, fluff_terms):
            continue
        if _TECHNICAL_LOG_PATTERNS.search(chunk):
            continue
        if _BACKGROUND_PATTERNS.search(chunk):
            continue
        cleaned = clean_text(chunk, fluff_terms)
        if not cleaned:
            continue
        cleaned = _LIST_MARKER.sub(" ", cleaned).strip()
        if len(cleaned.split()) < _MIN_SIGNAL_TOKENS:
            continue
        filtered.append(cleaned)
    return filtered


def prepare_signal_texts(text: str, nlp_model, fluff_terms: Iterable[str]) -> List[str]:
    candidates = chunk_notes(text, nlp_model)
    if not candidates:
        cleaned_full = clean_text(text, fluff_terms)
        return extract_signals_from_text(cleaned_full, nlp_model)
    filtered = _filter_signal_candidates(candidates, fluff_terms)
    return _deduplicate_signals(filtered)



def extract_signals_from_text(text: str, nlp_model) -> List[str]:
    if not text:
        return []
    if nlp_model is None:
        return [tok.strip() for tok in text.split() if len(tok.strip()) > 2]
    doc = nlp_model(text)
    signals = set()
    for chunk in doc.noun_chunks:
        normalized = chunk.text.strip()
        if len(normalized) > 2:
            signals.add(normalized)
    for token in doc:
        if not token.is_stop and token.is_alpha and len(token.text) > 2:
            signals.add(token.lemma_.lower())
    return sorted(signals)


def normalize_sentiment_label(label: str, score: float) -> str:
    label = label.upper()
    if label == "NEGATIVE":
        return "very_negative" if score > 0.7 else "negative"
    if label == "POSITIVE":
        return "very_positive" if score > 0.85 else "positive"
    return "neutral"


def aggregate_sentiment(score: float, thresholds: dict) -> str:
    if score >= thresholds.get("very_positive", 1.5):
        return "very_positive"
    if score >= thresholds.get("positive", 0.5):
        return "positive"
    if score >= thresholds.get("neutral", -0.5):
        return "neutral"
    if score >= thresholds.get("negative", -1.5):
        return "negative"
    return "very_negative"


def compute_signal_score(
    sentiment_label: str,
    sentiment_mapped_score: float,
    source: str,
    config: dict,
) -> float:
    source_weight = config.get("source_weights", {}).get(source.lower(), 1.0)
    severity_weight = config.get("severity_weights", {}).get(sentiment_label, 1.0)
    return sentiment_mapped_score * source_weight * severity_weight


def process_interaction(
    row: pd.Series,
    flair: dict | None,
    nlp_model,
    sentiment_model,
    config: dict,
) -> List[Signal]:
    fluff_terms = (flair or {}).get("fluff_terms", [])
    signal_texts = prepare_signal_texts(row["notes"], nlp_model, fluff_terms)
    mapped_scores = config.get("sentiment_score_map", {
        "VERY_POSITIVE": 2,
        "POSITIVE": 1,
        "NEUTRAL": 0,
        "NEGATIVE": -1,
        "VERY_NEGATIVE": -2,
    })
    signals = []
    for idx, signal_text in enumerate(signal_texts):
        sentiment = sentiment_model(signal_text[:512])
        prediction = sentiment[0]
        sentiment_label = normalize_sentiment_label(prediction.get("label", "NEUTRAL"), prediction.get("score", 0))
        mapped_score = mapped_scores.get(prediction.get("label", "NEUTRAL"), 0)
        signal_score = compute_signal_score(sentiment_label, mapped_score, row["source"], config)
        signal_id = hashlib.sha256(f"{row['id']}-{signal_text}-{idx}".encode()).hexdigest()[:12]
        signals.append(
            Signal(
                signal_id=signal_id,
                signal=signal_text,
                weight=config.get("source_weights", {}).get(row["source"].lower(), 1.0),
                signal_score=signal_score,
                signal_sentiment=sentiment_label,
            )
        )
    if not signals:
        # keep one row indicating no signal
        signal_id = hashlib.sha256(f"{row['id']}-no-signal".encode()).hexdigest()[:12]
        signals.append(
            Signal(
                signal_id=signal_id,
                signal="[no signal detected]",
                weight=1.0,
                signal_score=0.0,
                signal_sentiment="neutral",
            )
        )
    return signals


def process_batch(
    source_path: Path,
    output_dir: Path,
    config_paths: dict,
    output_format: str = "parquet",
    nlp_model=_SENTINEL,
    sentiment_model=_SENTINEL,
) -> Path:
    config = {}
    for key, path in config_paths.items():
        config[key] = load_yaml(path)
    data = pd.read_csv(source_path, parse_dates=["date"], dtype=str, encoding="utf-8", encoding_errors="replace")
    validate_schema(data)
    nlp_model = build_spacy_model() if nlp_model is _SENTINEL else nlp_model
    sentiment_model = build_sentiment_pipeline() if sentiment_model is _SENTINEL else sentiment_model
    records = []
    for _, row in data.iterrows():
        signals = process_interaction(row, config.get("fluff"), nlp_model, sentiment_model, config.get("weights", {}))
        weights_cfg = config.get("weights", {})
        bias = weights_cfg.get("negativity_bias", 1.0)
        max_signals = weights_cfg.get("max_signals_per_interaction", len(signals))
        # Keep the most extreme signals (by absolute score) up to the configured cap
        scored = sorted(signals, key=lambda s: abs(s.signal_score), reverse=True)[:max_signals]
        biased = [s.signal_score * bias if s.signal_score < 0 else s.signal_score for s in scored]
        aggregated_score = sum(biased) / len(biased) if biased else 0.0
        aggregated_sentiment = aggregate_sentiment(aggregated_score, config.get("weights", {}).get("aggregated_sentiment_thresholds", {}))
        for signal in signals:
            records.append(
                {
                    "id": row["id"],
                    "signal_id": signal.signal_id,
                    "signal": signal.signal,
                    "weight": signal.weight,
                    "signal_score": signal.signal_score,
                    "signal_sentiment": signal.signal_sentiment,
                    "aggregated_score": aggregated_score,
                    "aggregated_sentiment": aggregated_sentiment,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"signals.{output_format}"
    frame = pd.DataFrame.from_records(records)
    frame.sort_values(["id", "signal"], inplace=True)
    if output_format == "csv":
        frame.to_csv(output_path, index=False)
    else:
        frame.to_parquet(output_path, index=False)
    LOGGER.info("wrote %s rows to %s", len(frame), output_path)
    return output_path
