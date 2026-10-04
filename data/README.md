# Data and model cache

Everything in this directory is **git-ignored** (only `.gitkeep` placeholders are
tracked). Re-fetch or re-generate the files below; they are not part of the
repository.

## `raw/` — training corpora

Consumed by `uv run web-attack-detector-build-dataset`.

| Path                          | Used for        | Notes                                                        |
|-------------------------------|-----------------|--------------------------------------------------------------|
| `raw/xss/XSS_dataset.csv`     | XSS + benign    | UTF-8 BOM; index column; `Sentence`/`Label`                   |
| `raw/sqli/SQLi_dataset1.csv`  | SQLi + benign   | `Sentence`/`Label`; **310 rows** carry junk labels            |
| `raw/sqli/SQLi_dataset2.csv`  | SQLi + benign   | Text column is `Query`, not `Sentence`; CRLF line endings     |
| `raw/csrf/synthetic_csrf.txt` | CSRF            | One payload per line, no header                               |
| `raw/csrf/gen_csrf_new.csv`   | CSRF            | Header `payload`; **contains unescaped quotes** (invalid CSV) |
| `raw/csrf/gen_csrf_old.csv`   | CSRF            | Header `payload`                                              |
| `raw/generic/payloads.csv`    | benign + XSS    | Header is `payload, class` (**leading space** on `class`), CRLF |

The CSRF sources are **attack-only** — there is no benign HTTP-request corpus.

## `external/` — pre-trained embeddings

| Path                                | Size   | Notes                                                    |
|-------------------------------------|--------|----------------------------------------------------------|
| `external/use_model/`               | 988 MB | TF-Hub Universal Sentence Encoder v4 (local SavedModel)   |
| `external/glove-wiki-gigaword-100/` | 129 MB | GloVe 100-d, word2vec text format, `.gz`                 |

### Missing: fastText

`fasttext-wiki-news-subwords-300` (**~2.2 GB**) is **not** cached here and is
required by the optional `uniembed` detector. Fetch it once with network access:

```bash
uv sync --extra web --extra uniembed
uv run python -c "import gensim.downloader as api; api.load('fasttext-wiki-news-subwords-300')"
```

The loader looks for the gensim cache layout
`external/fasttext-wiki-news-subwords-300/fasttext-wiki-news-subwords-300.vec.gz`.
If it is absent and `WEB_ATTACK_DETECTOR_OFFLINE` is unset, it is downloaded on
demand; otherwise you get an actionable `FileNotFoundError`.

## `processed/` — generated

| Path                                  | Notes                                    |
|---------------------------------------|------------------------------------------|
| `processed/web_attacks_4class.csv`    | Output of `web-attack-detector-build-dataset`; columns `payload`, `label`, `attack_type` |

## `legacy/` — superseded, kept for reference

`fig-data/`, `maam-data/` and `new-xss/` hold figures, matrices and outputs from
the earlier RandomForest/LogisticRegression pipeline that was replaced by the
DistilBERT hybrid. Nothing in `src/` reads them.