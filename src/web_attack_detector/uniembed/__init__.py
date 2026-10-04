"""Uniembed + MLP detector (optional ``uniembed`` extra).

Importing this package requires ``tensorflow_hub`` and ``gensim``; the heavy
modules are imported lazily so that ``import web_attack_detector`` stays cheap.
"""

from __future__ import annotations

__all__ = ["UniembedDetector", "load_embeddings"]
