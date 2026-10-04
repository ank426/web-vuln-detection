"""Rule-based feature extractors for the uniembed+MLP detector.

The three extractors below are a *different* feature space from
:mod:`web_attack_detector.features`: they emit 67 XSS, 51 CSRF and 39 SQLi
values (157 total) rather than 45. That width, and the exact ordering within
each vector, is baked into ``models/uniembed/mlp_xss_csrf_sqli_detector.pth``
(``layer1.weight`` is 1069x256 = 157 + 100 + 300 + 512), so the bodies below are
kept byte-for-byte from the original ``app.py``. Do not reorder or "tidy" them.

Requires the optional ``uniembed`` extra.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

import numpy as np
from bs4 import BeautifulSoup

# Measured widths of the extractors below. The original ``app.py`` declared
# 67/51/39, but the functions actually emit 68/52/37; only the sum (157) is
# load-bearing, because it is the sum that feeds ``MLP.layer1`` (1069 = 157 + 912).
XSS_RULE_BASED_SIZE = 68
CSRF_RULE_BASED_SIZE = 52
SQLI_RULE_BASED_SIZE = 37
RULE_FEATURES_TOTAL = XSS_RULE_BASED_SIZE + CSRF_RULE_BASED_SIZE + SQLI_RULE_BASED_SIZE


def extract_xss_68_features(payload: str) -> np.ndarray:
    """Return the 68 XSS rule features, order fixed by ``XSS_FEATURE_NAMES_68``."""
    features: dict[str, float] = {}

    # --- Pre-processing ---
    # The raw payload is used for both parsing and direct string analysis.
    html_content = payload
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        features["malformed"] = 0
    except Exception:
        soup = BeautifulSoup("", "html.parser")
        features["malformed"] = 1

    js_code = "".join(script.get_text(strip=True) for script in soup.find_all("script"))

    # Attempt to find a URL within the payload, otherwise use a default
    url_match = re.search(r'https?://[^\s\'"]+', html_content)
    url = url_match.group(0) if url_match else "http://example.com"
    parsed_url = urlparse(url)

    # Category 1: URL Features
    features["url_length"] = len(url)
    features["url_special_characters"] = len(re.findall(r"[^a-zA-Z0-9]", url))
    features["url_tag_script"] = 1 if re.search(r"<script", url, re.IGNORECASE) else 0
    features["url_tag_iframe"] = 1 if re.search(r"<iframe", url, re.IGNORECASE) else 0
    features["url_attr_src"] = 1 if re.search(r"src=", url, re.IGNORECASE) else 0
    features["url_event_onload"] = 1 if re.search(r"onload=", url, re.IGNORECASE) else 0
    features["url_event_onmouseover"] = 1 if re.search(r"onmouseover=", url, re.IGNORECASE) else 0
    features["url_cookie"] = 1 if "cookie" in url.lower() else 0
    features["url_number_keywords_param"] = len(
        re.findall(r"(alert|script|onerror|onload|eval)", parsed_url.query, re.IGNORECASE)
    )
    features["url_number_domain"] = (
        len(parsed_url.hostname.split(".")) if parsed_url.hostname else 0
    )

    # Category 2: HTML Tag Features (from parsed payload)
    tags = [
        "script",
        "iframe",
        "meta",
        "object",
        "embed",
        "link",
        "svg",
        "frame",
        "form",
        "div",
        "style",
        "img",
        "input",
        "textarea",
    ]
    for tag in tags:
        features[f"html_tag_{tag}"] = len(soup.find_all(tag))

    # Category 3: HTML Attribute Features (from parsed payload)
    attrs = [
        "action",
        "background",
        "classid",
        "codebase",
        "href",
        "longdesc",
        "profile",
        "src",
        "usemap",
    ]
    all_tags = soup.find_all(True)
    for attr in attrs:
        features[f"html_attr_{attr}"] = sum(1 for tag in all_tags if attr in tag.attrs)
    features["html_attr_http-equiv"] = sum(1 for tag in all_tags if "http-equiv" in tag.attrs)

    # Category 4: HTML Event Handler Features (from parsed payload)
    events = [
        "onblur",
        "onchange",
        "onclick",
        "onerror",
        "onfocus",
        "onkeydown",
        "onkeypress",
        "onkeyup",
        "onload",
        "onmousedown",
        "onmouseout",
        "onmouseover",
        "onmouseup",
        "onsubmit",
    ]
    for event in events:
        features[f"html_event_{event}"] = sum(1 for tag in all_tags if event in tag.attrs)

    # Category 5: JavaScript & Content Features (from raw payload and extracted JS)
    evil_keywords = [
        "eval",
        "alert",
        "prompt",
        "confirm",
        "document.cookie",
        "window.location",
        "unescape",
    ]
    features["html_number_keywords_evil"] = sum(
        html_content.lower().count(kw) for kw in evil_keywords
    )
    features["js_file"] = len(soup.find_all("script", src=True))
    features["js_pseudo_protocol"] = len(re.findall(r"javascript:", html_content, re.IGNORECASE))
    features["js_dom_location"] = js_code.lower().count("location")
    features["js_dom_document"] = js_code.lower().count("document")
    features["js_prop_cookie"] = js_code.lower().count(".cookie")
    features["js_prop_referrer"] = js_code.lower().count(".referrer")

    js_methods = [
        "write",
        "getElementsByTagName",
        "getElementById",
        "alert",
        "eval",
        "fromCharCode",
        "confirm",
    ]
    for method in js_methods:
        features[f"js_method_{method}"] = len(
            re.findall(rf"[\b\.]({method})\s*\(", js_code, re.IGNORECASE)
        )

    features["js_min_length"] = len(js_code)
    features["js_min_define_function"] = len(re.findall(r"function\s*[\w]*\s*\(", js_code))
    total_calls = len(re.findall(r"\w+\s*\(", js_code))
    features["js_min_function_calls"] = max(0, total_calls - features["js_min_define_function"])
    js_strings = re.findall(r'["\'](.*?)["\']', js_code)
    features["js_string_max_length"] = max(len(s) for s in js_strings) if js_strings else 0

    # Category 6: Final General Feature
    features["html_length"] = len(html_content)

    # Return as a numpy array in a consistent order
    feature_order = sorted(features.keys())
    return np.array([features[k] for k in feature_order])


def extract_rule_based_csrf_features(payload: str) -> np.ndarray:
    """Return the 52 CSRF rule features, order fixed by ``CSRF_FEATURE_NAMES_RULE``."""
    feature_names = [
        "numOfParams",
        "numOfBools",
        "numOfIds",
        "numOfBlobs",
        "reqLen",
        "createInPath",
        "createInParams",
        "addInPath",
        "addInParams",
        "setInPath",
        "setInParams",
        "deleteInPath",
        "deleteInParams",
        "updateInPath",
        "updateInParams",
        "removeInPath",
        "removeInParams",
        "friendInPath",
        "friendInParams",
        "settingInPath",
        "settingInParams",
        "passwordInPath",
        "passwordInParams",
        "tokenInPath",
        "tokenInParams",
        "changeInPath",
        "changeInParams",
        "actionInPath",
        "actionInParams",
        "payInPath",
        "payInParams",
        "loginInPath",
        "loginInParams",
        "logoutInPath",
        "logoutInParams",
        "postInPath",
        "postInParams",
        "commentInPath",
        "commentInParams",
        "followInPath",
        "followInParams",
        "subscribeInPath",
        "subscribeInParams",
        "signInPath",
        "signInParams",
        "viewInPath",
        "viewInParams",
        "isPUT",
        "isDELETE",
        "isPOST",
        "isGET",
        "isOPTIONS",
    ]
    keywords = [
        "create",
        "add",
        "set",
        "delete",
        "update",
        "remove",
        "friend",
        "setting",
        "password",
        "token",
        "change",
        "action",
        "pay",
        "login",
        "logout",
        "post",
        "comment",
        "follow",
        "subscribe",
        "sign",
        "view",
    ]
    features = dict.fromkeys(feature_names, 0)
    payload_lower = payload.lower()
    features["reqLen"] = len(payload)
    url_match = re.search(r"https?://[^\'\"`\s<>]+", payload)
    path = ""
    all_params = {}
    if url_match:
        url = url_match.group(0)
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        all_params.update(parse_qs(parsed_url.query))
    form_param_match = re.search(r"send\('([^']+)'\)", payload)
    if form_param_match:
        all_params.update(parse_qs(form_param_match.group(1)))
    form_inputs = re.findall(
        r"<input[^>]+name=['\"]([^'\"]+)['\"][^>]+value=['\"]([^'\"]+)['\"]", payload
    )
    for name, value in form_inputs:
        if name not in all_params:
            all_params[name] = []
        all_params[name].append(value)
    features["numOfParams"] = len(all_params)
    for key, values in all_params.items():
        if "id" in key.lower():
            features["numOfIds"] += 1
        for value in values:
            if value.lower() in ["true", "false", "1", "0", "yes", "no"]:
                features["numOfBools"] += 1
    features["numOfBlobs"] = 0
    params_str = str(all_params.keys()).lower()
    for keyword in keywords:
        if keyword in path:
            features[f"{keyword}InPath"] = 1
        if keyword in params_str:
            features[f"{keyword}InParams"] = 1
    is_post = ("method" in payload_lower and "post" in payload_lower) or (
        "xhr.open('post'" in payload_lower
    )
    is_put = ("method" in payload_lower and "put" in payload_lower) or (
        "xhr.open('put'" in payload_lower
    )
    if is_post:
        features["isPOST"] = 1
    elif is_put:
        features["isPUT"] = 1
    elif (
        "method" in payload_lower and "delete" in payload_lower
    ) or "xhr.open('delete'" in payload_lower:
        features["isDELETE"] = 1
    elif (
        "method" in payload_lower and "options" in payload_lower
    ) or "xhr.open('options'" in payload_lower:
        features["isOPTIONS"] = 1
    else:
        features["isGET"] = 1
    return np.array([features[name] for name in feature_names])


def extract_sqli_features(payload: str) -> np.ndarray:
    """Return the 37 SQLi rule features, order fixed by ``SQLI_FEATURE_NAMES_RULE``."""
    features: list[float] = []
    payload_lower = payload.lower()

    # 1. SQL Keywords
    sql_keywords = [
        "select",
        "union",
        "insert",
        "update",
        "delete",
        "drop",
        "from",
        "where",
        "and",
        "or",
        "limit",
        "order by",
        "group by",
        "exec",
        "declare",
        "cast",
        "convert",
        "sleep",
        "benchmark",
        "waitfor",
    ]
    features.extend([payload_lower.count(kw) for kw in sql_keywords])

    # 2. SQL Comment Characters
    features.append(payload.count("--"))
    features.append(payload.count("#"))
    features.append(payload.count("/*"))
    features.append(payload.count("*/"))

    # 3. Tautologies and Common Bypass Patterns
    tautologies = ["1=1", "'='", "or 1=1", "or '1'='1'"]
    features.extend([payload_lower.count(t) for t in tautologies])

    # 4. Special Characters and Operators
    special_chars = ["=", "'", '"', ";", "(", ")"]
    features.extend([payload.count(char) for char in special_chars])

    # 5. Hex Encoded Characters
    features.append(len(re.findall(r"%[0-9a-f]{2}", payload_lower)))

    # 6. Overall Payload Length
    features.append(len(payload))

    # 7. Whitespace characters (often used for obfuscation)
    features.append(len(re.findall(r"\\s", payload)))

    return np.array(features, dtype=np.float32)
