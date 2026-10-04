"""Rule-based feature extraction for XSS, CSRF and SQLi payloads.

The three extractors below emit a *fixed-width, fixed-order* vector. That layout
is part of the model checkpoint contract -- ``NUM_RULE_FEATURES`` (45) is baked
into ``rule_processor.0.weight`` -- so the ordering here must not change without
retraining every checkpoint.

The ordering is made explicit via ``XSS_FEATURE_NAMES`` / ``CSRF_FEATURE_NAMES``
/ ``SQLI_FEATURE_NAMES`` rather than relying on ``dict`` insertion order, and is
covered by tests.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from urllib.parse import parse_qs, urlparse

import numpy as np
from bs4 import BeautifulSoup

DANGEROUS_TAGS: tuple[str, ...] = ("script", "iframe", "object", "embed", "svg", "form")
EVENT_HANDLERS: tuple[str, ...] = ("onclick", "onload", "onerror", "onmouseover")
JS_KEYWORDS: tuple[str, ...] = ("eval", "alert", "document.cookie", "window.location")
SENSITIVE_KEYWORDS: tuple[str, ...] = (
    "password",
    "token",
    "login",
    "delete",
    "update",
    "create",
)
HTTP_METHODS: tuple[str, ...] = ("post", "put", "delete", "get")
SQL_KEYWORDS: tuple[str, ...] = (
    "select",
    "union",
    "insert",
    "update",
    "delete",
    "drop",
    "where",
    "and",
    "or",
    "exec",
    "sleep",
    "benchmark",
)
INJECTION_PATTERNS: tuple[str, ...] = ("1=1", "'='", "or 1=1", "' or '1'='1'")
SPECIAL_CHARS: tuple[str, ...] = ("=", "'", '"', ";", "(", ")")

_URL_RE = re.compile(r"https?://[^\s'\"`]+")
_URL_WITH_QUERY_RE = re.compile(r"https?://[^\'\"`\s<>]+")
_FALLBACK_URL = "http://example.com"

XSS_FEATURE_NAMES: tuple[str, ...] = (
    "malformed",
    "url_length",
    "url_special_chars",
    "url_script_tag",
    "url_iframe_tag",
    *(f"html_{tag}_count" for tag in DANGEROUS_TAGS),
    *(f"event_{handler}" for handler in EVENT_HANDLERS),
    "js_keywords",
    "js_length",
)

CSRF_FEATURE_NAMES: tuple[str, ...] = (
    "request_length",
    "num_params",
    *(f"{keyword}_in_request" for keyword in SENSITIVE_KEYWORDS),
    *(f"is_{method}" for method in HTTP_METHODS),
)

SQLI_FEATURE_NAMES: tuple[str, ...] = (
    *(f"sql_{keyword}" for keyword in SQL_KEYWORDS),
    "sql_comments",
    "injection_patterns",
    "special_chars",
    "payload_length",
)


def _parse_html(payload: str) -> tuple[BeautifulSoup, float]:
    """Parse a payload as HTML, flagging failure rather than raising.

    ``html.parser`` is extremely permissive and effectively never raises, but a
    caller-supplied payload is untrusted input, so the failure flag is kept.
    """
    try:
        return BeautifulSoup(payload, "html.parser"), 0.0
    except Exception:
        return BeautifulSoup("", "html.parser"), 1.0


def extract_xss_features(payload: str) -> np.ndarray:
    """Extract the 17 XSS-specific features."""
    soup, malformed = _parse_html(payload)
    js_code = "".join(script.get_text(strip=True) for script in soup.find_all("script"))

    url_match = _URL_RE.search(payload)
    url = url_match.group(0) if url_match else _FALLBACK_URL

    features: list[float] = [
        malformed,
        float(len(url)),
        float(len(re.findall(r"[^a-zA-Z0-9]", url))),
        float(bool(re.search(r"<script", url, re.IGNORECASE))),
        float(bool(re.search(r"<iframe", url, re.IGNORECASE))),
    ]
    features.extend(float(len(soup.find_all(tag))) for tag in DANGEROUS_TAGS)
    all_tags = soup.find_all(True)
    features.extend(
        float(sum(1 for tag in all_tags if handler in tag.attrs)) for handler in EVENT_HANDLERS
    )
    payload_lower = payload.lower()
    features.append(float(sum(payload_lower.count(kw) for kw in JS_KEYWORDS)))
    features.append(float(len(js_code)))

    return np.asarray(features, dtype=np.float32)


def extract_csrf_features(payload: str) -> np.ndarray:
    """Extract the 12 CSRF-specific features."""
    payload_lower = payload.lower()

    features: list[float] = [float(len(payload))]

    url_match = _URL_WITH_QUERY_RE.search(payload)
    params: dict[str, list[str]] = {}
    if url_match:
        params = parse_qs(urlparse(url_match.group(0)).query)
    features.append(float(len(params)))

    features.extend(float(keyword in payload_lower) for keyword in SENSITIVE_KEYWORDS)
    features.extend(float(method in payload_lower) for method in HTTP_METHODS)

    return np.asarray(features, dtype=np.float32)


def extract_sqli_features(payload: str) -> np.ndarray:
    """Extract the 16 SQLi-specific features."""
    payload_lower = payload.lower()

    features: list[float] = [float(payload_lower.count(kw)) for kw in SQL_KEYWORDS]
    features.append(
        float(payload.count("--") + payload.count("#") + payload.count("/*")),
    )
    features.append(
        float(sum(payload_lower.count(pattern) for pattern in INJECTION_PATTERNS)),
    )
    features.append(float(sum(payload.count(char) for char in SPECIAL_CHARS)))
    features.append(float(len(payload)))

    return np.asarray(features, dtype=np.float32)


def extract_rule_features(payload: str) -> np.ndarray:
    """Concatenate all rule features into the ``NUM_RULE_FEATURES``-wide vector."""
    combined: np.ndarray = np.hstack(
        [
            extract_xss_features(payload),
            extract_csrf_features(payload),
            extract_sqli_features(payload),
        ],
    )
    return combined.astype(np.float32)


def extract_rule_features_batch(payloads: Sequence[str]) -> np.ndarray:
    """Vectorised :func:`extract_rule_features` over many payloads."""
    if not payloads:
        return np.zeros((0, NUM_RULE_FEATURES_TOTAL), dtype=np.float32)
    stacked = np.vstack([extract_rule_features(str(p)) for p in payloads])
    return stacked.astype(np.float32)


NUM_RULE_FEATURES_TOTAL = sum(
    (
        len(XSS_FEATURE_NAMES),
        len(CSRF_FEATURE_NAMES),
        len(SQLI_FEATURE_NAMES),
    ),
)
"""Runtime-computed total width; asserted against ``config.NUM_RULE_FEATURES`` in tests."""
