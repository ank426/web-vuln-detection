# XSS / CSRF / SQLi payload detector

A hybrid web-attack classifier: **DistilBERT** embeddings fused with **hand-crafted
rule features**, trained over a unified four-class corpus.

| Label | Class    | Detection target                                  |
|-------|----------|---------------------------------------------------|
| `0`   | `benign` | ordinary HTTP requests and non-malicious markup    |
| `1`   | `xss`    | reflected/stored script injection payloads         |
| `2`   | `csrf`   | cross-site request forgery payloads                |
| `3`   | `sqli`   | SQL injection payloads                             |

The label encoding is **frozen**. Changing it invalidates every shipped checkpoint.

## Layout

```
src/web_attack_detector/
  config.py        frozen constants, repo-relative paths
  features.py      the 45-dim rule branch (17 XSS + 12 CSRF + 16 SQLi)
  model.py         DistilBERT + rule-feature fusion trunk
  dataset.py       label validation + tokenised torch Dataset
  trainer.py       training, evaluation, inference, checkpoint I/O
  protocols.py     structural types for untyped third-party objects
  data/builder.py  unified corpus builder (all raw sources -> one CSV)
  apps/            Streamlit UIs
  uniembed/        optional GloVe + fastText + USE + MLP detector
  cli/             console entry points
data/              corpora and caches (git-ignored, see data/README.md)
models/            checkpoints (git-ignored)
notebooks/         original research notebooks
```

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra web          # DistilBERT detector + Streamlit UI
uv sync --extra web --extra uniembed   # also the secondary uniembed detector
```

Python 3.12 is pinned by `.python-version`.

## Build the dataset

Raw corpora are **not** committed. Once they are in place (see `data/README.md`),
build the unified training set:

```bash
uv run web-attack-detector-build-dataset --max-per-class 8000
```

This reads every raw source, tolerates the inconsistencies between them (mixed
encodings, BOMs, CRLF, differing text-column names, unescaped quotes inside
generated CSRF files), drops junk labels and empty payloads, de-duplicates,
optionally caps each class, and writes:

```
data/processed/web_attacks_4class.csv   # columns: payload, label, attack_type
```

## Train

```bash
uv run web-attack-detector-train --epochs 3 --batch-size 8
```

## Predict

```bash
uv run web-attack-detector-predict --payload "<script>alert(1)</script>"
uv run web-attack-detector-predict --payload "..." --json
```

## Web UI

```bash
uv run web-attack-detector-web
```

## Secondary detector (optional)

The `uniembed` extra provides the original GloVe/fastText/USE + MLP detector.
It needs the fastText corpus, which is not cached in this repository:

```bash
uv run python -c "import gensim.downloader as api; api.load('fasttext-wiki-news-subwords-300')"
uv run web-attack-detector-predict-uniembed --payload "..."
```

Set `WEB_ATTACK_DETECTOR_OFFLINE=1` to fail fast instead of downloading.

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```

`pre-commit install` wires all four into `.git/hooks/pre-commit`.

## Caveats

- The CSRF corpus is **attack-only**; there is no benign HTTP-request corpus.
  The model may learn that any form or XHR is CSRF. The builder warns about this.
