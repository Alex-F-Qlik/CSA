"""Tests for the sentiment signal pipeline."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import (
    aggregate_sentiment,
    chunk_notes,
    clean_text,
    compute_signal_score,
    extract_signals_from_text,
    process_batch,
    process_interaction,
)


class DummySentimentModel:
    def __call__(self, text: str) -> list[dict]:
        if "slow" in text or "error" in text:
            return [{"label": "NEGATIVE", "score": 0.8}]
        return [{"label": "POSITIVE", "score": 0.95}]


def test_clean_text_removes_fluff_terms():
    text = "This is really a very good and basically excellent product."
    cleaned = clean_text(text, ["really", "very", "basically"])
    assert "really" not in cleaned
    assert "basically" not in cleaned


def test_extract_signals_from_text_without_model():
    signals = extract_signals_from_text("awesome reliability great performance", nlp_model=None)
    assert "awesome" in signals
    assert "reliability" in signals


def test_compute_signal_score_respects_weights():
    score = compute_signal_score(
        sentiment_label="positive",
        sentiment_mapped_score=1,
        source="email",
        config={"source_weights": {"email": 1.25}, "severity_weights": {"positive": 2}},
    )
    assert score == 2.5


def test_aggregate_sentiment_thresholds():
    thresholds = {"very_positive": 2.0, "positive": 1.0, "neutral": 0.0, "negative": -1.0}
    assert aggregate_sentiment(2.5, thresholds) == "very_positive"
    assert aggregate_sentiment(0.5, thresholds) == "neutral"


def test_process_interaction_emits_default_signal():
    row = pd.Series(
        {
            "id": "10",
            "source": "email",
            "customer": "Acme",
            "date": "2026-03-01",
            "agent": "alice",
            "service_product": "Data Analytics",
            "program": "Managed Insights",
            "incident_id": "",
            "activity": "dashboard",
            "notes": "",
        }
    )
    signals = process_interaction(row, None, None, DummySentimentModel(), {})
    assert len(signals) == 1
    assert signals[0].signal == "[no signal detected]"
    assert signals[0].signal_sentiment == "neutral"


def test_process_batch_creates_normalized_output(tmp_path: Path):
    batch = tmp_path / "batch.csv"
    batch.write_text(
        "id,source,customer,date,agent,service_product,program,incident_id,activity,notes\n"
        "1,email,Acme,2026-03-01,alice,Data Analytics,Managed Insights,,dashboard,Great support is super helpful.\n"
        "2,ticket,Globex,2026-03-02,bob,Data Integration,Integration Care,INC-1,pipeline,Errors happen when loading data."
    )
    output_dir = tmp_path / "out"
    config_paths = {"fluff": Path("configs/fluff.yaml"), "weights": Path("configs/weights.yaml")}

    result_path = process_batch(
        source_path=batch,
        output_dir=output_dir,
        config_paths=config_paths,
        output_format="csv",
        nlp_model=None,
        sentiment_model=DummySentimentModel(),
    )
    assert result_path.exists()
    frame = pd.read_csv(result_path)
    assert {"id", "signal", "aggregated_score", "aggregated_sentiment"}.issubset(frame.columns)
    assert frame["aggregated_score"].astype(float).notna().all()


def test_chunk_notes_splits_on_conjunctions():
    text = "Great insights but data refresh is too slow and feels clunky."
    chunks = chunk_notes(text, nlp_model=None)
    assert len(chunks) == 2
    assert "great insights" in chunks[0].lower()
    assert "data refresh" in chunks[1].lower()


def test_process_interaction_chunks_notes_into_signals():
    row = pd.Series(
        {
            "id": "20",
            "source": "email",
            "customer": "Beta",
            "date": "2026-03-01",
            "agent": "alice",
            "service_product": "Data Analytics",
            "program": "Managed Insights",
            "incident_id": "",
            "activity": "dashboard",
            "notes": "Great insights but data refresh is too slow and feels clunky.",
        }
    )
    signals = process_interaction(row, None, None, DummySentimentModel(), {})
    assert len(signals) == 2
    texts = [signal.signal.lower() for signal in signals]
    assert any("great insights" in text for text in texts)
    assert any("data refresh" in text for text in texts)


def test_process_interaction_skips_greeting_only_note():
    row = pd.Series(
        {
            "id": "99",
            "source": "email",
            "customer": "Omega",
            "date": "2026-03-01",
            "agent": "alice",
            "service_product": "Data Analytics",
            "program": "Managed Insights",
            "incident_id": "",
            "activity": "dashboard",
            "notes": "Hi, thanks for attending the call.",
        }
    )
    signals = process_interaction(row, None, None, DummySentimentModel(), {})
    assert len(signals) == 1
    assert signals[0].signal == "[no signal detected]"
