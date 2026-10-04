"""Structural types for third-party objects we only use through duck typing.

Declaring these as :class:`~typing.Protocol` keeps the annotations meaningful
without forcing an import of the heavyweight optional dependencies
(``tensorflow_hub``, ``gensim``) on every code path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np


class Tokenizer(Protocol):
    """The slice of a HuggingFace fast tokenizer this package relies on."""

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        truncation: bool = ...,
        padding: str = ...,
        max_length: int = ...,
        return_tensors: str = ...,
    ) -> Mapping[str, object]:
        """Tokenise ``text`` into a mapping of batched tensor fields."""


class TfTensor(Protocol):
    """Structural view of a ``tf.Tensor``."""

    def numpy(self) -> np.ndarray:
        """Copy the tensor out of graph/ device memory as a NumPy array."""


class SentenceEncoder(Protocol):
    """Structural view of a TF-Hub Universal Sentence Encoder."""

    def __call__(self, text: Sequence[str]) -> TfTensor:
        """Encode a batch of strings into unit-norm vectors."""

    def __getitem__(self, key: object) -> np.ndarray:
        """Index into the underlying Keras model (used to reach sub-layers)."""
