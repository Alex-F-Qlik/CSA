# Customer Sentiment Signal Pipeline

This project ingests normalized customer feedback batches, extracts and qualifies "signals", applies configurable scoring, and emits normalized signal-level outputs suitable for downstream analysis.

## Features
- Batch ingestion of normalized feedback tables (CSV/Parquet).
- Configurable fluff removal, signal qualification, and sentiment scoring powered by spaCy and Hugging Face transformers.
- Config-driven weight maps for signal sources and severities.
- Outputs normalized schema with signal-level and aggregated scores/sentiments.
- Tests and CI to validate ingestion, scoring, and persistence logic.

## Setup
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running the pipeline
```
python -m src.cli data/sample_feedback.csv --output-dir output --format parquet
```

## Testing
```
pytest
```

## Configs
- `configs/fluff.yaml`: fluff terms filtered from free-form notes.
- `configs/weights.yaml`: source weights, severity weights, and sentiment thresholds.
