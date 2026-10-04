# web-vuln-detection

Classify web attack payloads as **benign**, **XSS**, **CSRF** or **SQLi**.

The model is a hybrid: **DistilBERT** embeddings fused with **hand-crafted rule
features**, which keeps the semantic power of a transformer while retaining the
precision of explicit signatures for injection syntax.

## Classes

| Label | Class    | Detection target                                 |
|-------|----------|--------------------------------------------------|
| `0`   | `benign` | ordinary HTTP requests and non-malicious markup   |
| `1`   | `xss`    | reflected/stored script injection payloads        |
| `2`   | `csrf`   | cross-site request forgery payloads               |
| `3`   | `sqli`   | SQL injection payloads                            |

The label encoding is **frozen**. Changing it invalidates every bundled checkpoint.

## How it works

```
payload
   │
   ├──► DistilBERT ──► 768-d ──┐
   │                            ├─► fusion MLP ──► 4 logits
   └──► rule features ─► 45-d ─┘
                              (17 XSS + 12 CSRF + 16 SQLi)
```

Both branches are necessary: the transformer generalises to obfuscated variants,
while the rule branch pins down exact syntax like `' OR 1=1 --` that a
subword tokenizer can easily blur.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned by `.python-version`.

```bash
uv sync --extra web                              # detector + Streamlit UI
uv sync --extra web --extra uniembed             # also the secondary model

uv run web-attack-detector-build-dataset --max-per-class 8000
uv run web-attack-detector-train --epochs 3 --batch-size 8
uv run web-attack-detector-predict --payload "<script>alert(1)</script>"
uv run web-attack-detector-web                   # Streamlit UI on :8501
```

## Dataset

Raw corpora are **not** committed; see [`data/README.md`](data/README.md) for the
provenance of every file. The builder reads all of them into one canonical CSV:

```
data/processed/web_attacks_4class.csv     # payload, label, attack_type
```

It tolerates the inconsistencies between the real sources — mixed encodings,
BOMs, CRLF, differing text-column names, unescaped quotes inside the generated
CSRF files — and reports what it discarded:

| Stage                | Rows     |
|----------------------|----------|
| Combined raw         | 88,375   |
| benign               | 52,113   |
| xss                  | 22,745   |
| sqli                 | 11,322   |
| csrf                 |  2,195   |
| junk labels dropped  |    310   |
| empty payloads       |    235   |
| duplicates removed   | 52,752   |
| **after `--max-per-class 8000`** | **26,195** |

## Layout

```
src/web_attack_detector/
  config.py        frozen constants, repo-relative paths
  features.py      the 45-dim rule branch
  model.py         DistilBERT + rule-feature fusion trunk
  dataset.py       label validation + tokenised torch Dataset
  trainer.py       training, evaluation, inference, checkpoint I/O
  protocols.py     structural types for untyped third-party objects
  data/builder.py  unified corpus builder
  apps/            Streamlit UIs
  uniembed/        optional GloVe + fastText + USE + MLP detector
  cli/             console entry points
data/              corpora and caches (git-ignored)
models/            checkpoints (git-ignored)
notebooks/         original research notebooks
tests/             86 tests, incl. checkpoint-compatibility checks
```

## Development

```bash
uv run ruff format .
uv run ruff check .
uv run ty check
uv run pytest
```

`pre-commit install` wires all four into `.git/hooks/pre-commit`.

## Secondary model (optional)

The `uniembed` extra provides the earlier GloVe + fastText + USE + MLP detector.
It needs the ~2.2 GB fastText corpus, which is not bundled:

```bash
uv run python -c "import gensim.downloader as api; api.load('fasttext-wiki-news-subwords-300')"
uv run web-attack-detector-predict-uniembed --payload "..."
```

Set `WEB_ATTACK_DETECTOR_OFFLINE=1` to fail fast instead of downloading.

## Limitations

- **CSRF is attack-only.** There is no benign HTTP-request corpus, so the model
  can learn that any form or XHR is CSRF. The builder emits a warning.
- **Rule-feature order is a contract.** The 45 rule features are positional; the
  existing tests assert this against the checkpoint weights.

## License

MIT — see [`LICENSE`](LICENSE).