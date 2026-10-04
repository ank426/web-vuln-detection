"""Tests for the rule-based feature extractors."""

from __future__ import annotations

import numpy as np
import pytest

from web_attack_detector.config import NUM_RULE_FEATURES
from web_attack_detector.features import (
    CSRF_FEATURE_NAMES,
    NUM_RULE_FEATURES_TOTAL,
    SQLI_FEATURE_NAMES,
    XSS_FEATURE_NAMES,
    extract_csrf_features,
    extract_rule_features,
    extract_rule_features_batch,
    extract_sqli_features,
    extract_xss_features,
)

PAYLOADS = [
    "",
    "test",
    "GET /index.php?id=1",
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert(1)>",
    "<svg onload=alert('XSS')>",
    "POST /login HTTP/1.1\nContent-Type: application/x-www-form-urlencoded\n\nuser=a&password=b",
    "' OR '1'='1' --",
    "1' UNION SELECT username,password FROM users--",
    "'; DROP TABLE users; --",
    '<iframe src="javascript:alert(1)"></iframe>',
    "<form action='http://target/x' method='POST'><input name='a'></form>",
    "https://example.com/a?b=1&c=2",
    "<<<>>> not html at all %zz {{{",
    "ünïcödé pãyload ☠",
]


def test_declared_widths_match_runtime_widths() -> None:
    """The 45-dim checkpoint contract must match the name tables."""
    assert len(XSS_FEATURE_NAMES) == 17
    assert len(CSRF_FEATURE_NAMES) == 12
    assert len(SQLI_FEATURE_NAMES) == 16
    assert NUM_RULE_FEATURES_TOTAL == 45
    assert NUM_RULE_FEATURES == NUM_RULE_FEATURES_TOTAL


@pytest.mark.parametrize("payload", PAYLOADS)
def test_rule_vector_width_is_constant(payload: str) -> None:
    """Every payload yields exactly NUM_RULE_FEATURES float32 values."""
    vector = extract_rule_features(payload)
    assert vector.shape == (NUM_RULE_FEATURES,)
    assert vector.dtype == np.float32


@pytest.mark.parametrize("payload", PAYLOADS)
def test_components_are_non_negative_and_finite(payload: str) -> None:
    """Counts and lengths cannot be negative or NaN."""
    for vector in (
        extract_xss_features(payload),
        extract_csrf_features(payload),
        extract_sqli_features(payload),
    ):
        assert np.all(np.isfinite(vector))
        assert np.all(vector >= 0)


def test_extractors_are_pure_functions() -> None:
    """Repeated calls give identical output (no dict-ordering or state leakage)."""
    payload = "<script>alert(1)</script><form method=POST action=x></form> OR 1=1--"
    first = extract_rule_features(payload)
    for _ in range(5):
        np.testing.assert_array_equal(extract_rule_features(payload), first)


def test_xss_counts_script_tags_and_handlers() -> None:
    """Known payloads produce the expected XSS signals."""
    vector = extract_xss_features("<script>alert('X')</script><img src=x onerror=alert(1)>")
    by_name = dict(zip(XSS_FEATURE_NAMES, vector, strict=True))
    assert by_name["malformed"] == 0.0
    assert by_name["html_script_count"] == 1.0
    assert by_name["event_onerror"] == 1.0
    assert by_name["js_keywords"] >= 2.0  # 'alert' appears twice
    assert by_name["js_length"] > 0.0


def test_xss_uses_fallback_url_when_absent() -> None:
    """A payload with no URL still yields a full-width vector."""
    by_name = dict(zip(XSS_FEATURE_NAMES, extract_xss_features("no url here"), strict=True))
    assert by_name["url_length"] == len("http://example.com")


def test_csrf_counts_query_params_of_absolute_url() -> None:
    """num_params reflects the parsed query string when an absolute URL is present."""
    by_name = dict(
        zip(CSRF_FEATURE_NAMES, extract_csrf_features("GET https://h/p?a=1&b=2"), strict=True),
    )
    assert by_name["num_params"] == 2.0
    assert by_name["is_get"] == 1.0


def test_csrf_detects_sensitive_keyword_and_method() -> None:
    """Password keywords and HTTP verbs are flagged."""
    by_name = dict(
        zip(
            CSRF_FEATURE_NAMES,
            extract_csrf_features("POST /change_password?token=abc"),
            strict=True,
        ),
    )
    assert by_name["password_in_request"] == 1.0
    assert by_name["token_in_request"] == 1.0
    assert by_name["is_post"] == 1.0
    # "get" is not a substring of this payload, unlike a naive prefix match.
    assert by_name["is_get"] == 0.0
    # num_params only counts query params of an *absolute* URL; there is none here.
    assert by_name["num_params"] == 0.0


def test_sqli_counts_keywords_and_comments() -> None:
    """SQL keywords, comments and tautologies are counted."""
    by_name = dict(
        zip(SQLI_FEATURE_NAMES, extract_sqli_features("' OR 1=1 -- select * from t"), strict=True),
    )
    assert by_name["sql_select"] == 1.0
    assert by_name["sql_or"] == 1.0
    assert by_name["sql_comments"] == 1.0
    assert by_name["injection_patterns"] >= 2.0
    # Only '=' and "'" appear; the payload has no parentheses or semicolons.
    assert by_name["special_chars"] == 2.0


def test_sqli_empty_payload_is_all_zero() -> None:
    """The empty string has no signals."""
    np.testing.assert_array_equal(extract_sqli_features(""), np.zeros(16, dtype=np.float32))


def test_batch_matches_individual() -> None:
    """Batched extraction equals per-payload extraction."""
    subset = PAYLOADS[:6]
    batched = extract_rule_features_batch(subset)
    assert batched.shape == (len(subset), NUM_RULE_FEATURES)
    for index, payload in enumerate(subset):
        np.testing.assert_array_equal(batched[index], extract_rule_features(payload))


def test_batch_of_empty_list_has_correct_shape() -> None:
    """An empty batch is a (0, 45) array rather than an error."""
    assert extract_rule_features_batch([]).shape == (0, NUM_RULE_FEATURES)


def test_batch_coerces_non_string_payloads() -> None:
    """Non-string inputs are stringified, matching the trainer's behaviour."""
    assert extract_rule_features_batch([123, None]).shape == (2, NUM_RULE_FEATURES)  # ty: ignore[invalid-argument-type]  # deliberately wrong type: asserts runtime coercion
