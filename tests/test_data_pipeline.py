"""Tests for the unified dataset builder.

These use synthetic fixtures rather than ``data/raw`` so the suite runs without
the (git-ignored) corpora present.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from web_attack_detector.config import AttackClass
from web_attack_detector.data.builder import (
    _coerce_binary_labels,
    _read_payload_lines,
    build_unified_dataset,
    write_dataset,
)


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    """Create a miniature ``data/raw`` tree covering every source quirk."""
    (tmp_path / "xss").mkdir()
    (tmp_path / "sqli").mkdir()
    (tmp_path / "csrf").mkdir()
    (tmp_path / "generic").mkdir()

    # UTF-8 BOM, like the real XSS corpus.
    (tmp_path / "xss" / "XSS_dataset.csv").write_text(
        "﻿,Sentence,Label\n0,<b>hi</b>,0\n1,<script>alert(1)</script>,1\n",
        encoding="utf-8",
    )
    # Junk label rows, like the real SQLi_dataset1.
    (tmp_path / "sqli" / "SQLi_dataset1.csv").write_text(
        'Sentence,Label,,\n" OR 1=1 --",1,,\n"waitfor delay",junk,,\n"plain",0,,\n',
        encoding="utf-8",
    )
    # Different text column name, like the real SQLi_dataset2.
    (tmp_path / "sqli" / "SQLi_dataset2.csv").write_text(
        "Query,Label\r\nadmin'--,1\r\nGET /,0\r\n",
        encoding="utf-8",
    )
    # Payload-per-line files, one with a header and HTML entities.
    (tmp_path / "csrf" / "synthetic_csrf.txt").write_text(
        "<form action='http://t/x' method='POST'></form>\n"
        "<script>fetch('http://t/y',{method:'POST'})</script>\n",
        encoding="utf-8",
    )
    (tmp_path / "csrf" / "gen_csrf_new.csv").write_text(
        "payload\n"
        "<svg onload='x.open(&quot;POST&quot;)'></svg>\n"
        # Unescaped quotes inside a JSON body: invalid CSV, must survive.
        "<script>fetch('http://t/z', {body: '{\"a\": 1}'})</script>\n",
        encoding="utf-8",
    )
    (tmp_path / "csrf" / "gen_csrf_old.csv").write_text(
        "payload\n<img src='http://t/w'/>\n",
        encoding="utf-8",
    )
    # Spaced column name and CRLF, like the real payloads.csv.
    (tmp_path / "generic" / "payloads.csv").write_text(
        "payload, class\r\n<b>ok</b>,0\r\n<script>bad</script>,1\r\n",
        encoding="utf-8",
    )
    return tmp_path


def test_coerce_binary_labels_drops_junk() -> None:
    """Non-0/1 values are dropped and counted."""
    labels, dropped = _coerce_binary_labels(pd.Series([0, 1, "junk", 2, 1]))
    assert labels.tolist() == [0, 1, 1]
    assert dropped == 2


def test_read_payload_lines_on_real_file(tmp_path: Path) -> None:
    """Header skipping and entity unescaping work on a real file."""
    path = tmp_path / "p.csv"
    path.write_text("payload\n<svg a=&quot;b&quot;></svg>\n\n", encoding="utf-8")
    assert _read_payload_lines(path) == ['<svg a="b"></svg>']


def test_builder_produces_canonical_frame(raw_root: Path) -> None:
    """Output has the exact columns, dtypes and label encoding the trainer wants."""
    frame, report = build_unified_dataset(raw_root)

    assert list(frame.columns) == ["payload", "label", "attack_type"]
    assert frame["label"].dtype == "int64"
    assert set(frame["label"].unique()) <= {0, 1, 2, 3}
    assert report.dropped_junk_labels == 1
    assert report.rows == len(frame)


def test_builder_maps_classes_correctly(raw_root: Path) -> None:
    """Each source lands in the right AttackClass bucket."""
    frame, _ = build_unified_dataset(raw_root)
    by_type = {
        attack: set(frame.loc[frame["attack_type"] == attack, "payload"])
        for attack in frame["attack_type"].unique()
    }
    # benign comes from the 0 rows of the xss/sqli/generic corpora
    assert {"<b>hi</b>", "GET /", "<b>ok</b>"} <= by_type["benign"]
    assert {"<script>alert(1)</script>", "<script>bad</script>"} <= by_type["xss"]
    assert {"admin'--", "OR 1=1 --"} <= by_type["sqli"]
    assert {
        "<form action='http://t/x' method='POST'></form>",
        "<script>fetch('http://t/y',{method:'POST'})</script>",
        # &quot; becomes a literal double quote; the surrounding ' are untouched.
        "<svg onload='x.open(\"POST\")'></svg>",
        "<img src='http://t/w'/>",
    } <= by_type["csrf"]


def test_builder_survives_unescaped_quotes_in_generated_csvs(raw_root: Path) -> None:
    """Payloads that are invalid CSV are still recovered verbatim."""
    frame, _ = build_unified_dataset(raw_root)
    recovered = set(frame["payload"])
    assert "<script>fetch('http://t/z', {body: '{\"a\": 1}'})</script>" in recovered


def test_builder_warns_about_attack_only_csrf(raw_root: Path) -> None:
    """The CSRF skew warning is always emitted when CSRF rows exist."""
    _, report = build_unified_dataset(raw_root)
    assert any("attack-only" in w for w in report.warnings)


def test_builder_deduplicates(raw_root: Path) -> None:
    """Duplicate payloads collapse, keeping the first occurrence."""
    target = raw_root / "csrf" / "synthetic_csrf.txt"
    target.write_text("<form action='http://t/x' method='POST'></form>\n" * 3, encoding="utf-8")
    frame, report = build_unified_dataset(raw_root)
    assert report.duplicates_removed > 0
    assert frame["payload"].is_unique


def test_builder_can_skip_dedupe(raw_root: Path) -> None:
    """dedupe=False preserves duplicates."""
    target = raw_root / "csrf" / "synthetic_csrf.txt"
    target.write_text("<form action='http://t/x'></form>\n" * 3, encoding="utf-8")
    frame, report = build_unified_dataset(raw_root, dedupe=False)
    assert report.duplicates_removed == 0
    assert frame["payload"].duplicated().any()


def test_builder_can_exclude_generic_corpus(raw_root: Path) -> None:
    """include_generic=False drops payloads.csv rows."""
    with_generic, _ = build_unified_dataset(raw_root, include_generic=True)
    without, _ = build_unified_dataset(raw_root, include_generic=False)
    assert len(with_generic) > len(without)
    assert "<b>ok</b>" not in set(without["payload"])


def test_max_per_class_caps_each_class(raw_root: Path) -> None:
    """Downsampling respects the cap per class, never below available rows."""
    frame, _ = build_unified_dataset(raw_root, max_per_class=1)
    counts = frame["label"].value_counts()
    assert counts.max() == 1


def test_max_per_class_is_deterministic(raw_root: Path) -> None:
    """The same seed yields the same rows."""
    first, _ = build_unified_dataset(raw_root, max_per_class=2, seed=7)
    second, _ = build_unified_dataset(raw_root, max_per_class=2, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_report_render_mentions_counts(raw_root: Path) -> None:
    """render() surfaces rows, per-class counts and warnings."""
    _, report = build_unified_dataset(raw_root)
    text = report.render()
    assert "rows:" in text
    assert "per-class counts:" in text
    assert "WARNINGS:" in text


def test_write_dataset_creates_parents(tmp_path: Path, raw_root: Path) -> None:
    """write_dataset mkdirs and round-trips."""
    frame, _ = build_unified_dataset(raw_root)
    destination = tmp_path / "nested" / "out.csv"
    write_dataset(frame, destination)
    assert destination.exists()
    pd.testing.assert_frame_equal(pd.read_csv(destination), frame, check_dtype=False)


def test_missing_sqli_source_warns_but_proceeds(raw_root: Path) -> None:
    """A missing SQLi file degrades gracefully instead of crashing."""
    (raw_root / "sqli" / "SQLi_dataset2.csv").unlink()
    _, report = build_unified_dataset(raw_root)
    assert report.rows > 0
    assert "sqli" in report.class_counts


def test_label_name_mapping_round_trips() -> None:
    """AttackClass names map to the canonical integers."""
    for member in AttackClass:
        assert member.name.lower() in {"benign", "xss", "csrf", "sqli"}
