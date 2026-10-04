"""Training, evaluation, checkpointing and inference for :class:`HybridAttackDetector`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from web_attack_detector.config import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    NUM_CLASSES,
    NUM_RULE_FEATURES,
)
from web_attack_detector.dataset import WebAttackDataset, validate_labels
from web_attack_detector.features import extract_rule_features_batch
from web_attack_detector.model import HybridAttackDetector
from web_attack_detector.protocols import Tokenizer


@dataclass(frozen=True)
class Prediction:
    """A single prediction."""

    label: int
    class_name: str
    confidence: float
    probabilities: np.ndarray


@dataclass(frozen=True)
class EpochMetrics:
    """Loss and accuracy for one epoch."""

    loss: float
    accuracy: float


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate evaluation output."""

    loss: float
    accuracy: float
    predictions: np.ndarray
    labels: np.ndarray
    report: str


def resolve_device(device: str) -> torch.device:
    """Resolve a device string, honouring ``"auto"``."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class WebAttackTrainer:
    """Owns the tokenizer, model, optimiser and dataloaders for one training run.

    Args:
        model_name: HuggingFace encoder id or local path.
        num_classes: Number of output classes.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ...
        learning_rate: AdamW learning rate.
        max_length: Tokenisation length.
        batch_size: Default dataloader batch size.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        device: str = "auto",
        learning_rate: float = DEFAULT_LEARNING_RATE,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Bind the model, optimiser, tokenizer and device for a training run."""
        self.device = resolve_device(device)
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.batch_size = batch_size
        self.class_names: tuple[str, ...] = CLASS_NAMES[:num_classes]

        # from_pretrained is typed as possibly returning None (e.g. when an
        # architecture has no tokenizer); fail loudly now instead of at the
        # first forward pass.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer is None:
            msg = f"No tokenizer available for model {model_name!r}"
            raise RuntimeError(msg)
        self.tokenizer: Tokenizer = tokenizer
        self.model = HybridAttackDetector(
            model_name=model_name,
            num_classes=num_classes,
            rule_feature_size=NUM_RULE_FEATURES,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        self.criterion = nn.CrossEntropyLoss()

        self.train_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

    def extract_features(self, payloads: list[str]) -> torch.Tensor:
        """Extract rule features for a batch of payloads as a CPU tensor."""
        matrix = extract_rule_features_batch(payloads)
        return torch.from_numpy(matrix).to(torch.device("cpu"))

    def prepare_data(
        self,
        df: pd.DataFrame,
        payload_col: str,
        label_col: str,
        test_size: float = DEFAULT_TEST_SIZE,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> tuple[DataLoader, DataLoader]:
        """Validate a dataframe and build stratified train/test dataloaders.

        Args:
            df: Source frame containing the payload and label columns.
            payload_col: Name of the text column.
            label_col: Name of the integer label column.
            test_size: Fraction held out for evaluation.
            batch_size: Overrides the trainer default.
            max_length: Overrides the trainer default.

        Returns:
            The ``(train_loader, test_loader)`` pair.

        Raises:
            KeyError: If either column is missing.
            LabelSchemaError: If labels are not canonical integers.
        """
        missing = [c for c in (payload_col, label_col) if c not in df.columns]
        if missing:
            raise KeyError(
                f"Missing column(s) {missing}; available: {list(df.columns)}",
            )

        payloads = df[payload_col].astype(str).to_numpy()
        labels = validate_labels(df[label_col].to_numpy(), num_classes=self.num_classes)

        x_train, x_test, y_train, y_test = train_test_split(
            payloads,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=DEFAULT_SEED,
        )

        length = max_length or self.max_length
        effective_batch = batch_size or self.batch_size
        train_loader = DataLoader(
            WebAttackDataset(x_train, y_train, self.tokenizer, length),
            batch_size=effective_batch,
            shuffle=True,
        )
        test_loader = DataLoader(
            WebAttackDataset(x_test, y_test, self.tokenizer, length),
            batch_size=effective_batch,
            shuffle=False,
        )
        self.train_loader, self.test_loader = train_loader, test_loader
        return train_loader, test_loader

    def _require_loader(self, which: str) -> DataLoader:
        loader = self.train_loader if which == "train" else self.test_loader
        if loader is None:
            msg = "No dataloader; call prepare_data() first."
            raise RuntimeError(msg)
        return loader

    def _batch_rule_features(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.extract_features(list(batch["payload"])).to(self.device)

    def train_epoch(self) -> EpochMetrics:
        """Run one training epoch."""
        loader = self._require_loader("train")
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            rule_features = self._batch_rule_features(batch)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask, rule_features)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += float(loss.item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)

        return EpochMetrics(loss=total_loss / len(loader), accuracy=correct / total)

    def evaluate(self) -> EvaluationResult:
        """Evaluate on the held-out loader."""
        loader = self._require_loader("test")
        self.model.eval()
        total_loss = 0.0
        predictions: list[np.ndarray] = []
        labels: list[np.ndarray] = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["label"].to(self.device)
                rule_features = self._batch_rule_features(batch)

                logits = self.model(input_ids, attention_mask, rule_features)
                total_loss += float(self.criterion(logits, batch_labels).item())

                predictions.append(logits.argmax(dim=1).cpu().numpy())
                labels.append(batch_labels.cpu().numpy())

        all_predictions = np.concatenate(predictions) if predictions else np.empty(0, dtype=int)
        all_labels = np.concatenate(labels) if labels else np.empty(0, dtype=int)
        accuracy = float((all_predictions == all_labels).mean()) if all_labels.size else 0.0
        report = classification_report(
            all_labels,
            all_predictions,
            target_names=list(self.class_names),
            zero_division=0,
        )
        return EvaluationResult(
            loss=total_loss / len(loader),
            accuracy=accuracy,
            predictions=all_predictions,
            labels=all_labels,
            report=report,
        )

    def train(
        self,
        epochs: int,
        save_path: str | Path,
        config: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> Path:
        """Train for ``epochs``, checkpointing the best validation accuracy.

        Args:
            epochs: Number of passes over the training data.
            save_path: Checkpoint destination.
            config: Optional config dict written alongside the checkpoint.
            verbose: Print per-epoch progress.

        Returns:
            The checkpoint path.
        """
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        best_accuracy = -1.0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch()
            evaluation = self.evaluate()
            if verbose:
                print(f"Epoch {epoch}/{epochs}")
                print("-" * 50)
                print(
                    f"Train Loss: {train_metrics.loss:.4f}, "
                    f"Train Acc: {train_metrics.accuracy:.4f}",
                )
                print(
                    f"Val   Loss: {evaluation.loss:.4f}, Val Acc: {evaluation.accuracy:.4f}",
                )

            if evaluation.accuracy > best_accuracy:
                best_accuracy = evaluation.accuracy
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_accuracy": best_accuracy,
                        "class_names": list(self.class_names),
                        "model_name": self.model_name,
                        "max_length": self.max_length,
                        "num_rule_features": NUM_RULE_FEATURES,
                    },
                    destination,
                )
                if verbose:
                    print(f"New best model saved with accuracy: {best_accuracy:.4f}")

        if verbose:
            print("=" * 50)
            print("FINAL EVALUATION")
            print("=" * 50)
            print(f"Best Validation Accuracy: {best_accuracy:.4f}")
            print(evaluation.report)

        if config is not None:
            config_path = destination.parent / "config.json"
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return destination

    def predict(self, payload: str) -> Prediction:
        """Classify a single payload."""
        self.model.eval()
        encoding = self.tokenizer(
            payload,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        rule_features = torch.from_numpy(extract_rule_features_batch([payload])).to(self.device)
        input_ids = cast("torch.Tensor", encoding["input_ids"]).to(self.device)
        attention_mask = cast("torch.Tensor", encoding["attention_mask"]).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, rule_features)
            probabilities = functional.softmax(logits, dim=1)

        label = int(logits.argmax(dim=1).item())
        return Prediction(
            label=label,
            class_name=self.class_names[label],
            confidence=float(probabilities.max().item()),
            probabilities=probabilities.cpu().numpy().flatten(),
        )

    def predict_batch(self, payloads: list[str]) -> list[Prediction]:
        """Classify many payloads, one tokenizer pass each."""
        return [self.predict(p) for p in payloads]

    def load_model(self, model_path: str | Path, *, load_optimizer: bool = True) -> None:
        """Load a checkpoint written by :meth:`train`.

        Handles both the legacy ``_use_new_zipfile_serialization=False`` format
        used by earlier commits and the current default format.

        Args:
            model_path: Checkpoint to load.
            load_optimizer: Restore optimiser state. Disable for inference-only
                loads, which cannot restore a differently-shaped optimiser.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
            KeyError: If the checkpoint is missing ``model_state_dict``.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state = checkpoint.get("model_state_dict")
        if state is None:
            raise KeyError(f"Checkpoint {path} has no 'model_state_dict' key")

        self.model.load_state_dict(state)
        if load_optimizer and checkpoint.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.class_names = tuple(checkpoint.get("class_names", self.class_names))
        print(f"Model loaded from {path}")
