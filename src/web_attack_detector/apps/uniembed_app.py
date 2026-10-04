"""Streamlit UI for the optional uniembed + MLP detector.

This is the port of the original ``new/app.py``, with its hardcoded
``/kaggle/input/...`` paths replaced by :mod:`web_attack_detector.config`, its
stale feature-width constants corrected to the values the shipped checkpoint
actually expects (68 / 52 / 37), and its two placeholder tabs implemented.

Requires the ``uniembed`` extra and the fastText corpus; see ``data/README.md``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from web_attack_detector.config import CLASS_NAMES, models_dir
from web_attack_detector.uniembed.model import UniembedDetector

st.set_page_config(page_title="Web Attack Detector (uniembed)", page_icon="🛡️", layout="wide")

EXAMPLES: dict[str, list[str]] = {
    "XSS": [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert(1)>",
        "<svg/onload=alert(document.domain)>",
    ],
    "CSRF": [
        "<form action='https://bank.example/transfer' method='POST'>"
        "<input name='amount'><input name='password'></form>",
        "<script>fetch('https://bank.example/logout',{method:'POST'})</script>",
    ],
    "SQLi": [
        "' OR 1=1 --",
        "admin'--",
        "1; DROP TABLE users; --",
    ],
    "Benign": [
        "GET /index.html HTTP/1.1",
        "<p>Hello, world</p>",
        "https://example.com/search?q=python",
    ],
}

MAX_BATCH = 5_000


@st.cache_resource(show_spinner="Loading uniembed models…")
def load_detector(checkpoint: str) -> UniembedDetector:
    """Load the MLP and its embedding bundles once and reuse them across reruns."""
    return UniembedDetector(checkpoint)


def resolve_checkpoint() -> Path | None:
    """Return the uniembed checkpoint path if it exists."""
    path = models_dir() / "uniembed" / "mlp_xss_csrf_sqli_detector.pth"
    return path if path.exists() else None


def render_single(detector: UniembedDetector) -> None:
    """Render the single-payload analysis tab."""
    payload = st.text_area(
        "Payload",
        height=140,
        placeholder="<script>alert(1)</script>",
    )
    if not st.button("Classify", type="primary") or not payload.strip():
        return

    with st.spinner("Analysing…"):
        label, probabilities, confidence = detector.predict(payload)

    name = CLASS_NAMES[label]
    left, right = st.columns(2)
    with left:
        st.metric(f"Prediction: {name}", f"{confidence:.2%} confidence")
    with right:
        st.bar_chart(
            pd.DataFrame(
                {"class": list(CLASS_NAMES), "probability": [float(p) for p in probabilities]},
            ).set_index("class"),
        )


def render_batch(detector: UniembedDetector) -> None:
    """Render the batch analysis tab."""
    uploaded = st.file_uploader("Upload a .txt file (one payload per line)", type=["txt"])
    text = (
        uploaded.getvalue().decode("utf-8")
        if uploaded
        else st.text_area("Or paste payloads (one per line)", height=200)
    )
    if not st.button("Classify batch") or not text.strip():
        return

    payloads = [line.strip() for line in text.splitlines() if line.strip()]
    if len(payloads) > MAX_BATCH:
        st.warning(f"Truncating to the first {MAX_BATCH} of {len(payloads)} payloads.")
        payloads = payloads[:MAX_BATCH]

    rows: list[dict[str, object]] = []
    progress = st.progress(0.0, text="Classifying…")
    for index, payload in enumerate(payloads, start=1):
        label, _probabilities, confidence = detector.predict(payload)
        rows.append(
            {
                "#": index,
                "Payload": payload[:120] + ("…" if len(payload) > 120 else ""),
                "Prediction": CLASS_NAMES[label],
                "Confidence": confidence,
            },
        )
        progress.progress(index / len(payloads), text=f"Classifying {index}/{len(payloads)}")
    progress.empty()

    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.bar_chart(frame["Prediction"].value_counts())
    with st.expander("JSON"):
        st.json(rows)


def render_examples(detector: UniembedDetector) -> None:
    """Render the worked-examples tab."""
    category = st.selectbox("Category", list(EXAMPLES))
    for example in EXAMPLES[category]:
        left, right = st.columns([3, 1])
        left.code(example, language="html")
        if right.button("Classify", key=f"example-{example}"):
            label, _, confidence = detector.predict(example)
            st.success(f"**{CLASS_NAMES[label]}** — {confidence:.2%}")


def main() -> None:
    """Render the app."""
    st.title("🛡️ Web Attack Detector")
    st.caption("Uniembed (GloVe + fastText + USE) + MLP classifier.")

    checkpoint = resolve_checkpoint()
    if checkpoint is None:
        st.error(
            f"No uniembed checkpoint at {checkpoint}. It ships with the project "
            "but is git-ignored; restore it from models/uniembed/.",
        )
        return

    try:
        detector = load_detector(str(checkpoint))
    except (OSError, RuntimeError, FileNotFoundError) as exc:
        st.error(f"Failed to load {checkpoint.name}: {exc}")
        st.info(
            "The fastText corpus (~2.2 GB) is required and is not bundled. "
            "See `data/README.md` for the one-off download command.",
        )
        return

    st.caption(f"Model: `{checkpoint.name}`")

    tab_single, tab_batch, tab_examples = st.tabs(
        ["Single payload", "Batch", "Examples"],
    )
    with tab_single:
        render_single(detector)
    with tab_batch:
        render_batch(detector)
    with tab_examples:
        render_examples(detector)


main()
