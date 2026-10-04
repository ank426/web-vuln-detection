"""Uniembed + MLP detector: rule features concatenated with GloVe, fastText and USE.

Requires the optional ``uniembed`` extra (``uv sync --extra uniembed``).

Unlike the original ``app.py``, all model locations resolve through
:mod:`web_attack_detector.config` rather than hardcoded ``/kaggle/input/...``
paths. GloVe and fastText come from the ``gensim`` downloader (cached under
``data/external``); the Universal Sentence Encoder is loaded from
``data/external/use_model`` if present, else fetched from tfhub.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from web_attack_detector.config import NUM_CLASSES, external_dir
from web_attack_detector.protocols import SentenceEncoder

from .features import (
    RULE_FEATURES_TOTAL,
    extract_rule_based_csrf_features,
    extract_sqli_features,
    extract_xss_68_features,
)

GLOVE_KEY = "glove-wiki-gigaword-100"
FASTTEXT_KEY = "fasttext-wiki-news-subwords-300"
GLOVE_FILENAME = "glove-wiki-gigaword-100.gz"
FASTTEXT_FILENAME = "fasttext-wiki-news-subwords-300.vec.gz"
USE_TFHUB_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

WORD2VEC_DIM = 100
FASTTEXT_DIM = 300
USE_DIM = 512
EMBEDDING_DIMS = WORD2VEC_DIM + FASTTEXT_DIM + USE_DIM
INPUT_SIZE = RULE_FEATURES_TOTAL + EMBEDDING_DIMS
"""1069; matches ``layer1.weight`` in the shipped checkpoint."""


class MLP(nn.Module):
    """Three-hidden-layer MLP over the 1069-dim concatenated feature vector."""

    def __init__(self, input_size: int = INPUT_SIZE, num_classes: int = NUM_CLASSES) -> None:
        """Create the 1069 -> 256 -> 128 -> 64 -> num_classes head."""
        super().__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits."""
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        return self.output_layer(x)


@dataclass(frozen=True)
class EmbeddingBundle:
    """The three pre-trained embedding models plus their tokenizers.

    ``word2vec`` and ``fasttext`` are ``gensim`` KeyedVectors; ``use`` is a
    ``tensorflow_hub`` Keras model. Both live in the optional ``uniembed`` extra,
    so they are typed as opaque objects here.
    """

    word2vec: WordVectors
    fasttext: WordVectors
    use: SentenceEncoder


def load_embeddings(cache_dir: str | None = None) -> EmbeddingBundle:
    """Load GloVe, fastText and USE, preferring local copies over downloads.

    Args:
        cache_dir: Override for the download cache. Defaults to
            ``data/external``.

    Returns:
        The loaded models.
    """
    import tensorflow_hub as hub

    root = Path(cache_dir) if cache_dir is not None else external_dir()

    use_local = root / "use_model"
    use_model = hub.load(str(use_local)) if use_local.exists() else hub.load(USE_TFHUB_URL)

    return EmbeddingBundle(
        word2vec=_load_word_vectors(GLOVE_KEY, GLOVE_FILENAME, root),
        fasttext=_load_word_vectors(FASTTEXT_KEY, FASTTEXT_FILENAME, root),
        use=use_model,
    )


def _load_word_vectors(key: str, filename: str, root: Path) -> WordVectors:
    """Load a word2vec/fastText model, preferring a local cache over the network.

    The ``gensim`` downloader resolves its cache through
    ``gensim.downloader.base_dir``, which is read at *import* time, so redirecting
    it after the fact is unreliable. Loading the cached file directly avoids that
    entirely; the downloader is only consulted for a model we do not already have.
    """
    from gensim.models import KeyedVectors

    cached = root / key / filename
    if cached.exists():
        return KeyedVectors.load_word2vec_format(str(cached))

    if not _download_is_allowed(root):
        msg = (
            f"Missing embedding corpus {key!r}: expected {cached}. "
            f"Fetch it into that path (the gensim downloader cache layout), or "
            f"run with network access to download it automatically."
        )
        raise FileNotFoundError(msg)

    import gensim.downloader as api

    api.base_dir = str(root)
    return api.load(key)


def _download_is_allowed(root: Path) -> bool:
    """Report whether a missing corpus may be fetched from the gensim mirror."""
    del root
    return not os.environ.get("WEB_ATTACK_DETECTOR_OFFLINE")


class WordVectors(Protocol):
    """The slice of ``gensim`` KeyedVectors this module needs.

    Declared structurally so the annotation stays valid without the optional
    ``uniembed`` extra installed.
    """

    key_to_index: Mapping[str, int]

    def __getitem__(self, key: str) -> np.ndarray:
        """Return the vector stored for ``key``."""


def _mean_word_embedding(text: str, model: WordVectors, dim: int) -> np.ndarray:
    """Mean of in-vocabulary token vectors, or zeros when nothing is in vocab."""
    vectors = [model[token] for token in text.lower().split() if token in model.key_to_index]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim, dtype=np.float32)


def extract_embedding_features(payload: str, embeddings: EmbeddingBundle) -> np.ndarray:
    """Concatenate the 912-dim GloVe + fastText + USE block."""
    cleaned = re.sub(r"<[^>]+>", " ", payload).strip()
    word2vec = _mean_word_embedding(cleaned, embeddings.word2vec, WORD2VEC_DIM)
    fasttext = _mean_word_embedding(cleaned, embeddings.fasttext, FASTTEXT_DIM)
    use = embeddings.use([payload]).numpy().flatten()
    return np.hstack([word2vec, fasttext, use]).astype(np.float32)


def extract_full_feature_vector(payload: str, embeddings: EmbeddingBundle) -> np.ndarray:
    """Return the 1069-dim vector: 157 rule features then 912 embedding features.

    The ordering here is the checkpoint contract.
    """
    rule = np.hstack(
        [
            extract_xss_68_features(payload),
            extract_rule_based_csrf_features(payload),
            extract_sqli_features(payload),
        ],
    )
    return np.hstack([rule, extract_embedding_features(payload, embeddings)]).astype(np.float32)


class UniembedDetector:
    """Loads the MLP checkpoint and classifies payloads.

    Args:
        checkpoint: Path to ``mlp_xss_csrf_sqli_detector.pth``.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``.
        embeddings: Pre-loaded embeddings; loaded on demand when omitted.
    """

    def __init__(
        self,
        checkpoint: str,
        device: str = "auto",
        embeddings: EmbeddingBundle | None = None,
    ) -> None:
        """Load the uniembed MLP and the embedding bundles it consumes."""
        self.device = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device),
        )
        self.embeddings = embeddings if embeddings is not None else load_embeddings()
        self.model = MLP().to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict(self, payload: str) -> tuple[int, np.ndarray, float]:
        """Classify one payload.

        Returns:
            ``(label, probabilities, confidence)``.
        """
        vector = extract_full_feature_vector(payload, self.embeddings)
        tensor = torch.from_numpy(vector).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probabilities = functional.softmax(logits, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
        return (
            int(prediction.cpu().item()),
            probabilities.cpu().numpy().flatten(),
            float(confidence.cpu().item()),
        )
