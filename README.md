# Customer Sentiment Signal Pipeline

This project ingests normalized customer feedback batches, extracts and qualifies "signals", applies configurable scoring, and emits normalized signal-level outputs suitable for downstream analysis.

## Features
- Batch ingestion of normalized feedback tables (CSV/Parquet), with UTF-8 fault tolerance for files saved from Windows applications.
- Multi-layer signal qualification: greeting/social filter, background narrative filter, technical log filter, minimum token gate, and near-duplicate suppression.
- Three-layer PII and boilerplate handling: line-level pre-processing strips email headers, distribution lists, and ITSM template blocks; PII masking replaces email addresses and person names before signal text is stored; a boilerplate fragment filter catches residual noise at the sentence level.
- Asymmetric, ITSM-aligned scoring: negative signals carry greater weight than positive, with a configurable negativity bias and mean-based aggregation capped to the most significant signals per interaction.
- Config-driven weight maps for signal sources, severity, negativity bias, and aggregation thresholds (`configs/weights.yaml`).
- Configurable fluff term removal using word-boundary matching (`configs/fluff.yaml`).
- Outputs normalized schema with signal-level and aggregated scores/sentiments.

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

### `configs/fluff.yaml`
List of filler words removed from signal text before scoring (e.g. `really`, `basically`, `hi`). Terms are matched on word boundaries to avoid corrupting words that contain them.

### `configs/weights.yaml`
| Key | Purpose |
|---|---|
| `source_weights` | Score multiplier by interaction channel (`email`, `ticket`, `escalation`, etc.) |
| `severity_weights` | Score multiplier by sentiment level. Negative labels carry more weight than positive by design. |
| `negativity_bias` | Additional multiplier applied to negative scores before aggregation (default: `1.5`). |
| `max_signals_per_interaction` | Cap on how many signals (by absolute score) contribute to the aggregated score (default: `7`). |
| `sentiment_score_map` | Maps sentiment labels to base numeric scores before weights are applied. |
| `aggregated_sentiment_thresholds` | Score bands used to classify the final aggregated sentiment per interaction. |

## Progress & Roadmap
See [progress.md](progress.md) for a detailed log of what has been implemented and known gaps.
