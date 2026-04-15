# Pipeline Progress

## Implemented

### Signal Qualification
- **Conjunction-based chunking** — spaCy splits sentences, then each chunk is further split on adversative conjunctions (`but`, `however`, `although`, etc.) to isolate contrasting sentiments within a single sentence.
- **Greeting / social courtesy filter** — chunks that open with greeting prefixes (`hi`, `hello`, `thanks for`, `hopefully`, etc.) are dropped before scoring. Checks original text before fluff removal so that phrases like "Thanks for your support" are caught even after the word "thanks" is stripped.
- **Minimum token gate** — chunks shorter than 5 meaningful tokens are discarded.
- **Background narrative filter (`_BACKGROUND_PATTERNS`)** — blacklist of sentence patterns that represent timeline narration, reproduction steps, test descriptions, or routing boilerplate (e.g. `"we also tested"`, `"it was only through"`, `"if you have any questions about"`, `"please contact the servicedesk"`). These carry no customer judgment and are dropped.
- **Technical log filter (`_TECHNICAL_LOG_PATTERNS`)** — drops lines containing SQL error fields (`RetCode:`, `SqlState:`), ODBC/SQL Server tags, and SQL error message bodies (`cannot insert`).
- **Neutral intent gate (`_NEUTRAL_INTENT_PATTERNS`)** — sentences that are informational statements or access requests (e.g. `"Note: we already have..."`, `"we are requesting"`, `"can you please"`) are kept in the output for context but their score is forced to `0.0 / neutral` rather than run through the sentiment model. This prevents the model from misreading factual request language as negative.
- **Near-duplicate suppression** — after filtering, signals sharing ≥ 60% of non-stopword vocabulary (Jaccard similarity) are collapsed; the first-encountered is kept.

### PII & Boilerplate Handling (Layer 1–3)
- **Layer 1 — Pre-processing (`_preprocess_notes`)** — line-by-line pass before spaCy sees the text:
  - Email thread header lines (`From:`, `To:`, `Sent:`, `Subject:`, `CC:`) are removed.
  - Distribution list lines containing ≥ 2 role/org markers (`contractor`, `consultant`, `accenture`, etc.) are removed.
  - ITSM / ServiceNow template lines and email security boilerplate (`assignment group`, `do not click links`, `caution this email originated`, `please do not reply`, `task\d+ sd_\w+` system queue references, etc.) are removed.
- **Layer 2 — PII masking (`_mask_pii`)** — applied after pre-processing, before chunking:
  - Email addresses replaced with `[EMAIL]` via regex.
  - Person names replaced with `[PERSON]` via spaCy NER (processed right-to-left to preserve character offsets).
- **Layer 3 — Boilerplate fragments in `_BACKGROUND_PATTERNS`** — catches courtesy/routing phrases that survive line-level removal (`"if you have any questions about"`, `"please contact the servicedesk"`, etc.).

### Text Cleaning
- **Word-boundary fluff removal** — fluff terms in `configs/fluff.yaml` are removed using `\b` word boundaries, preventing substring corruption (e.g. `"hi"` previously mangled `"this"` → `"t s"`).
- **Numbered list marker stripping** — leading/trailing list numbers (`1.`, `2)`, etc.) are removed from cleaned signal text.

### Scoring & Aggregation
- **Asymmetric severity weights** — negative signals carry higher weight than positive ones (`very_negative: 2.25`, `negative: 1.5` vs `very_positive: 1.5`), reflecting the ITSM industry standard that complaints have greater impact than equivalent praise.
- **Negativity bias** — negative signal scores are multiplied by a configurable bias factor (default `1.5`) before aggregation, preventing a single compliment from neutralizing a serious complaint.
- **Mean-based aggregation with signal cap** — aggregated score is the mean of the top N signals by absolute score (configurable via `max_signals_per_interaction`, default 7), replacing the previous summation which inflated scores linearly with signal count.
- **Recalibrated aggregation thresholds** — thresholds in `configs/weights.yaml` are calibrated for the mean-based scoring range.

### Robustness
- **UTF-8 encoding fallback** — CSV reads use `encoding_errors="replace"` to handle Windows-1252 special characters (em dash, curly quotes) without crashing.

---

## Known Gaps / Left to Implement

### Signal Quality
- **`_DIST_LIST_MARKERS` is vendor-specific** — the distribution list detector currently looks for `accenture`, `bluo software`, `qlik consultant`, etc. It will miss distribution lists from other customer environments. A more robust approach would detect the structural pattern (alternating name–role–email triplets) rather than vendor names.
- **`_BOILERPLATE_LINE` and `_NEUTRAL_INTENT_PATTERNS` are not exhaustive** — each new customer ITSM system (Jira, Zendesk, Freshservice, SAP) has its own template language, and new request phrasings will be encountered over time. These pattern lists will grow as new sources are onboarded.
- **Numbered list items currently kept as separate signals** — items 1–4 in a formal bug report produce 4 signals that are related. They should ideally be grouped under their parent complaint rather than scored independently.
- **No cross-interaction deduplication** — if a customer submits the same complaint in two tickets, both produce signals. There is no cross-row deduplication.

### Sentiment Model
- **Default model is generic** — the Hugging Face `sentiment-analysis` default (`distilbert-base-uncased-finetuned-sst-2-english`) is trained on movie reviews, not B2B technical support language. Domain-specific fine-tuning on ITSM text would improve accuracy, especially for neutral-sounding technical complaints.
- **Binary label only** — the base model outputs POSITIVE/NEGATIVE only; `normalize_sentiment_label()` infers `very_positive`/`very_negative` from the confidence score. A model with multi-class output (`VERY_NEGATIVE`, `NEGATIVE`, `NEUTRAL`, `POSITIVE`, `VERY_POSITIVE`) would be more accurate.
- **512-token truncation** — long signals are silently truncated at 512 tokens before scoring. No warning is logged and the truncated portion is unscored.

### PII
- **spaCy `en_core_web_sm` NER has limited person recall** — the small model misses names in non-standard formats (e.g. `LAST, First` or names in ALL CAPS). A larger model (`en_core_web_lg`) or a dedicated NER model would improve coverage.
- **No phone number or ID masking** — ticket IDs, phone numbers, and account numbers are not currently masked.
- **PII masking is not reversible** — `[PERSON]` / `[EMAIL]` replacements are applied directly to the stored signal text with no mapping back to the original. If re-identification is ever needed for audit purposes, a tokenization vault would be required.

### Pipeline Operability
- **No input validation on notes field** — `notes` values that are `NaN`, numeric, or non-string will cause a runtime error in `prepare_signal_texts`. Should be coerced/logged before processing.
- **No per-row error isolation** — if one row fails (e.g. malformed note), the entire batch fails. Rows should be processed in a try/except with the failing row logged and skipped.
- **No progress logging for large batches** — for batches with hundreds of rows, there is no indication of how far processing has progressed.
- **Output always overwrites `signals.csv/parquet`** — running the pipeline twice on different batches overwrites the previous output. Consider timestamped output filenames or an append mode.
