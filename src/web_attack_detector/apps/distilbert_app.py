"""Streamlit UI for the DistilBERT + rule-feature detector."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from web_attack_detector.config import default_model_path, models_dir

st.set_page_config(page_title="Web Attack Detector", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
def load_trainer(model_path: str) -> object:
    """Load the checkpoint once and reuse it across reruns."""
    import torch

    from web_attack_detector.trainer import WebAttackTrainer

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    class_names = checkpoint.get("class_names") or ["benign", "xss", "csrf", "sqli"]
    trainer = WebAttackTrainer(
        model_name=checkpoint.get("model_name") or "distilbert-base-uncased",
        num_classes=len(class_names),
        max_length=int(checkpoint.get("max_length") or 512),
    )
    trainer.load_model(model_path, load_optimizer=False)
    return trainer


def resolve_model_path() -> Path | None:
    """Return the first available checkpoint, preferring ``config.json`` siblings."""
    candidates = [default_model_path()]
    config = models_dir() / "config.json"
    if config.exists():
        try:
            saved = json.loads(config.read_text(encoding="utf-8"))
            if saved.get("model_path"):
                candidates.insert(0, Path(saved["model_path"]))
        except (OSError, json.JSONDecodeError):
            pass
    return next((c for c in candidates if c.exists()), None)


def render_results(trainer: object, payloads: list[str]) -> pd.DataFrame:
    """Classify each payload and render a results table."""
    rows: list[dict[str, object]] = []
    progress = st.progress(0.0, text="Classifying…")
    for index, payload in enumerate(payloads, start=1):
        prediction = trainer.predict(payload)  # ty: ignore[unresolved-attribute]
        rows.append(
            {
                "#": index,
                "Payload": payload[:120] + ("…" if len(payload) > 120 else ""),
                "Prediction": prediction.class_name,
                "Confidence": prediction.confidence,
            },
        )
        progress.progress(index / len(payloads), text=f"Classifying {index}/{len(payloads)}")
    progress.empty()
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True, hide_index=True)

    counts = frame["Prediction"].value_counts()
    st.bar_chart(counts)

    with st.expander("JSON"):
        st.json(rows)
    return frame


def main() -> None:
    """Render the app."""
    st.title("🛡️ Web Attack Detector")
    st.caption("DistilBERT embeddings fused with rule-based XSS / CSRF / SQLi features.")

    model_path = resolve_model_path()
    if model_path is None:
        st.error(
            f"No checkpoint found at {default_model_path()}. Train one with "
            "`uv run web-attack-detector-train`.",
        )
        return

    try:
        trainer = load_trainer(str(model_path))
    except (OSError, RuntimeError, KeyError) as exc:
        st.error(f"Failed to load {model_path}: {exc}")
        return

    st.caption(f"Model: `{model_path.name}`")

    tab_single, tab_batch = st.tabs(["Single payload", "Batch"])

    with tab_single:
        payload = st.text_area("Payload", height=140, placeholder="<script>alert(1)</script>")
        if st.button("Classify", type="primary") and payload.strip():
            prediction = trainer.predict(payload)  # ty: ignore[unresolved-attribute]
            st.metric(
                f"Prediction: {prediction.class_name}",
                f"{prediction.confidence:.2%} confidence",
            )
            st.bar_chart(
                pd.DataFrame(
                    {
                        "class": list(trainer.class_names),  # ty: ignore[unresolved-attribute]
                        "probability": [float(p) for p in prediction.probabilities],
                    },
                ).set_index("class"),
            )

    with tab_batch:
        uploaded = st.file_uploader("Upload a .txt file (one payload per line)", type=["txt"])
        text = (
            uploaded.getvalue().decode("utf-8")
            if uploaded
            else st.text_area(
                "Or paste payloads (one per line)",
                height=200,
            )
        )
        if st.button("Classify batch") and text.strip():
            payloads = [line.strip() for line in text.splitlines() if line.strip()]
            if len(payloads) > 5000:
                st.warning(f"Truncating to the first 5000 of {len(payloads)} payloads.")
                payloads = payloads[:5000]
            render_results(trainer, payloads)


if __name__ == "__main__":
    main()
else:  # pragma: no cover - streamlit executes scripts as __main__
    main()
