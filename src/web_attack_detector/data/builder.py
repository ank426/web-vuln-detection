"""Build the unified 4-class training set from the raw sources in ``data/raw``.

Every raw source in this project is *binary* (``0``/``1``) or *unlabelled*; the
detector needs a single 4-class frame with columns ``payload``, ``label`` and
``attack_type``. This module performs that reconciliation and reports on the
resulting class balance.

Two data-quality problems are corrected here rather than being pushed onto the
trainer:

* ``SQLi_dataset1.csv`` has ~310 rows whose ``Label`` cell contains a leaked SQL
  fragment instead of a class (rows shifted during collection). They are dropped
  and counted.
* Several sources carry a UTF-8 BOM, CRLF endings, or inconsistently named text
  columns (``Sentence`` vs ``Query``).

CSRF is *attack-only*: the workspace contains CSRF attack payloads but no
equivalent benign HTTP-request corpus. :func:`build_unified_dataset` emits an
explicit warning for this, since a model trained on it will tend to learn
"any form or XHR is CSRF".
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from html import unescape
from pathlib import Path

import numpy as np
import pandas as pd

from web_attack_detector.config import AttackClass, raw_dir
from web_attack_detector.sample_data import replicate_for_smoke_test

ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "latin-1")
"""Encoding fallbacks, tried in order; ``utf-8-sig`` strips the BOM."""


@dataclass
class DatasetReport:
    """Summary of a build, including anything the caller should know about."""

    rows: int
    class_counts: dict[str, int]
    dropped_junk_labels: int = 0
    dropped_empty: int = 0
    duplicates_removed: int = 0
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = [
            f"rows: {self.rows}",
            "per-class counts:",
            *(f"  {name:<8} {count:>8,}" for name, count in sorted(self.class_counts.items())),
            f"dropped junk labels:    {self.dropped_junk_labels:,}",
            f"dropped empty payloads: {self.dropped_empty:,}",
            f"duplicates removed:     {self.duplicates_removed:,}",
        ]
        if self.warnings:
            lines.append("WARNINGS:")
            lines.extend(f"  ! {w}" for w in self.warnings)
        return "\n".join(lines)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV, trying each encoding until one parses cleanly.

    Column names are whitespace-stripped: ``payloads.csv`` ships a header of
    ``payload, class`` with a leading space on the second field, and a BOM on
    some files otherwise survives as part of the first name.
    """
    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            frame = pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        frame.columns = [str(c).strip().lstrip("﻿") for c in frame.columns]
        return frame
    msg = f"Could not decode {path} with any of {ENCODINGS}: {last_error}"
    raise ValueError(msg) from last_error


def _first_present(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """Return the first candidate column that exists in ``df``."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    msg = f"None of {list(candidates)} found in columns {list(df.columns)}"
    raise KeyError(msg)


def _coerce_binary_labels(
    raw: pd.Series,
) -> tuple[pd.Series, int]:
    """Coerce a label column to ``int8`` 0/1, dropping rows that will not coerce.

    Args:
        raw: The raw label column.

    Returns:
        ``(labels, dropped_count)`` where ``labels`` is aligned to the rows that
        survived coercion.
    """
    numeric = pd.to_numeric(raw, errors="coerce")
    keep = numeric.isin([0, 1])
    dropped = int((~keep).sum())
    return numeric[keep].astype("int8"), dropped


def load_xss_corpus(root: Path | None = None) -> tuple[pd.DataFrame, int]:
    """Load the binary XSS corpus.

    Returns:
        ``(frame, dropped)`` with ``payload`` and binary ``label`` columns. Label
        ``0`` is benign and ``1`` is XSS.
    """
    path = (root or raw_dir()) / "xss" / "XSS_dataset.csv"
    df = _read_csv(path)
    text_col = _first_present(df, ("Sentence", "sentence", "payload"))
    labels, dropped = _coerce_binary_labels(df["Label"])
    return pd.DataFrame(
        {"payload": df.loc[labels.index, text_col].astype(str), "label": labels}
    ), dropped


def load_sqli_corpora(root: Path | None = None) -> tuple[pd.DataFrame, int]:
    """Load and concatenate both binary SQLi corpora.

    Handles the differing text column names (``Sentence`` / ``Query``) and drops
    rows whose label cell holds a leaked SQL fragment instead of ``0``/``1``.

    Returns:
        ``(frame, dropped)``.
    """
    base = (root or raw_dir()) / "sqli"
    frames: list[pd.DataFrame] = []
    dropped = 0
    for name in ("SQLi_dataset1.csv", "SQLi_dataset2.csv"):
        path = base / name
        if not path.exists():
            warnings.warn(f"Missing SQLi source {path}; skipping", stacklevel=2)
            continue
        df = _read_csv(path)
        text_col = _first_present(df, ("Sentence", "Query", "sentence", "query"))
        labels, dropped_here = _coerce_binary_labels(df["Label"])
        dropped += dropped_here
        frames.append(
            pd.DataFrame({"payload": df.loc[labels.index, text_col].astype(str), "label": labels}),
        )
    if not frames:
        msg = f"No SQLi sources found under {base}"
        raise FileNotFoundError(msg)
    return pd.concat(frames, ignore_index=True), dropped


def load_generic_payloads(root: Path | None = None) -> pd.DataFrame:
    """Load the large binary ``payload``/``class`` corpus (XSS vs benign)."""
    path = (root or raw_dir()) / "generic" / "payloads.csv"
    df = _read_csv(path)
    text_col = _first_present(df, ("payload", "Payload", "Sentence"))
    label_col = _first_present(df, ("class", "Class", "Label", "label"))
    labels, _ = _coerce_binary_labels(df[label_col])
    return pd.DataFrame({"payload": df.loc[labels.index, text_col].astype(str), "label": labels})


def _read_payload_lines(path: Path) -> list[str]:
    """Read a newline-delimited payload file, skipping a ``payload`` header.

    The generated CSRF files carry a ``payload`` header but are not valid CSV:
    several rows embed unescaped double quotes inside a JSON body, so a CSV
    parser raises ``ParserError``. Each file is really one payload per line, so
    they are read as plain text and HTML-unescaped (the generator wrote ``&quot;``
    where a literal quote belonged).
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].strip().lower() in {"payload", "sentence", "query"}:
        lines = lines[1:]
    return [unescape(line.strip()) for line in lines if line.strip()]


def load_csrf_attacks(root: Path | None = None) -> pd.DataFrame:
    """Load all unlabelled CSRF attack payloads.

    Combines the synthetic corpus with the two generated corpora. Every row is an
    attack by construction, so the returned frame has ``label`` fixed to ``1``.

    Returns:
        A frame whose ``label`` is all ones.
    """
    base = (root or raw_dir()) / "csrf"
    payloads: list[str] = []

    for name in ("synthetic_csrf.txt", "gen_csrf_new.csv", "gen_csrf_old.csv"):
        path = base / name
        if not path.exists():
            warnings.warn(f"Missing CSRF source {path}; skipping", stacklevel=2)
            continue
        payloads.extend(_read_payload_lines(path))

    return pd.DataFrame({"payload": payloads, "label": [1] * len(payloads)}).astype(
        {"label": "int8"},
    )


def _relabel(frame: pd.DataFrame, *, negative: int, positive: int) -> pd.DataFrame:
    """Map a binary frame's ``0``/``1`` onto two target class labels."""
    out = frame.copy()
    out["label"] = out["label"].map({0: negative, 1: positive}).astype("int64")
    return out


def _attack_type(series: pd.Series) -> pd.Series:
    """Map integer labels to readable class names."""
    return series.map(lambda v: AttackClass(int(v)).name.lower())


def _max_per_class(df: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    """Downsample each class to at most ``cap`` rows, deterministically.

    Sampling is driven by a single seeded generator over each group's own index,
    so the result depends only on ``(cap, seed)`` and the input row order.
    """
    if cap is None:
        return df
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby("label", sort=True):
        count = min(cap, len(group))
        chosen = rng.choice(group.index.to_numpy(), size=count, replace=False)
        parts.append(group.loc[np.sort(chosen)])
    return pd.concat(parts).sort_index()


def build_unified_dataset(
    root: Path | None = None,
    *,
    max_per_class: int | None = None,
    dedupe: bool = True,
    seed: int = 42,
    include_generic: bool = True,
) -> tuple[pd.DataFrame, DatasetReport]:
    """Assemble the unified 4-class training set.

    Args:
        root: Raw data root; defaults to ``data/raw``.
        max_per_class: Optionally cap each class at this many rows.
        dedupe: Drop duplicate payloads, keeping the first occurrence.
        seed: Seed for downsampling.
        include_generic: Include the large ``payloads.csv`` benign/XSS corpus.

    Returns:
        ``(frame, report)`` where the frame has ``payload``, ``label`` and
        ``attack_type`` columns.
    """
    report = DatasetReport(rows=0, class_counts={})

    xss_df, _ = load_xss_corpus(root)
    sqli_df, junk = load_sqli_corpora(root)
    csrf_df = load_csrf_attacks(root)
    report.dropped_junk_labels = junk

    benign = AttackClass.BENIGN
    xss = AttackClass.XSS
    csrf = AttackClass.CSRF
    sqli = AttackClass.SQLI

    parts = [
        # XSS corpus provides both benign (0) and XSS (1) rows.
        _relabel(xss_df, negative=int(benign), positive=int(xss)),
        # SQLi corpora provide both benign (0) and SQLi (1) rows.
        _relabel(sqli_df, negative=int(benign), positive=int(sqli)),
        # CSRF is attack-only.
        _relabel(csrf_df, negative=int(benign), positive=int(csrf)),
    ]
    if include_generic:
        generic = load_generic_payloads(root)
        parts.append(_relabel(generic, negative=int(benign), positive=int(xss)))

    combined = pd.concat(parts, ignore_index=True)
    combined["payload"] = combined["payload"].str.strip()

    empty = combined["payload"].eq("")
    report.dropped_empty = int(empty.sum())
    combined = combined.loc[~empty]

    if dedupe:
        pre_dedupe = len(combined)
        combined = combined.drop_duplicates(subset="payload", keep="first")
        report.duplicates_removed = pre_dedupe - len(combined)

    combined = _max_per_class(combined, max_per_class, seed).reset_index(drop=True)
    combined["label"] = combined["label"].astype("int64")
    combined["attack_type"] = _attack_type(combined["label"])

    report.rows = len(combined)
    counts = combined["label"].value_counts().to_dict()
    report.class_counts = {
        AttackClass(int(label)).name.lower(): int(count) for label, count in counts.items()
    }

    present = set(report.class_counts)
    missing = {c.name.lower() for c in AttackClass} - present
    if missing:
        report.warnings.append(f"no rows for class(es): {sorted(missing)}")

    min_count = min(report.class_counts.values()) if report.class_counts else 0
    max_count = max(report.class_counts.values()) if report.class_counts else 0
    if max_count and min_count and max_count / min_count > _IMBALANCE_RATIO:
        report.warnings.append(
            f"class imbalance {max_count / min_count:.1f}x between the largest and "
            f"smallest class; consider --max-per-class",
        )
    if report.class_counts.get(AttackClass.CSRF.name.lower()):
        report.warnings.append(
            "CSRF rows are attack-only: this project has no benign HTTP-request "
            "corpus, so the model may learn that any form/XHR is CSRF",
        )

    columns = ["payload", "label", "attack_type"]
    return combined[columns], report


_IMBALANCE_RATIO = 10.0


def build_smoke_dataset(repeats: int = 50) -> pd.DataFrame:
    """Return the replicated synthetic frame for the zero-data smoke path."""
    from web_attack_detector.sample_data import create_sample_data

    return replicate_for_smoke_test(create_sample_data(), repeats)


def write_dataset(frame: pd.DataFrame, destination: Path) -> Path:
    """Write ``frame`` to ``destination`` as CSV, creating parent directories."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def iter_sources(root: Path | None = None) -> Iterable[tuple[Path, str]]:
    """Yield ``(path, note)`` for each raw source, for manifest generation."""
    base = root or raw_dir()
    notes = {
        "xss/XSS_dataset.csv": "binary benign/XSS text; UTF-8 BOM",
        "sqli/SQLi_dataset1.csv": "binary benign/SQLi text; ~310 junk label rows",
        "sqli/SQLi_dataset2.csv": "binary benign/SQLi text; text column is 'Query'",
        "csrf/synthetic_csrf.txt": "unlabelled CSRF attacks, one per line",
        "csrf/gen_csrf_new.csv": "unlabelled CSRF attacks, 'payload' column",
        "csrf/gen_csrf_old.csv": "unlabelled CSRF attacks, 'payload' column",
        "generic/payloads.csv": "binary benign/XSS text; 64k rows; column is 'class'",
    }
    for rel, note in notes.items():
        yield base / rel, note
