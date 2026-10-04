"""Hybrid detector: DistilBERT embeddings fused with rule-based features."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from web_attack_detector.config import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MODEL_NAME,
    NUM_CLASSES,
    NUM_RULE_FEATURES,
)


class HybridAttackDetector(nn.Module):
    """Mean-pooled transformer encoder fused with a rule-feature MLP branch.

    Args:
        model_name: HuggingFace encoder id or local path.
        num_classes: Size of the classification head.
        rule_feature_size: Width of the rule-feature vector. Must equal
            :data:`~web_attack_detector.config.NUM_RULE_FEATURES` for checkpoints
            trained by this project to load.
        hidden_size: Width of the fusion/classifier trunk.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        rule_feature_size: int = NUM_RULE_FEATURES,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
    ) -> None:
        """Build the DistilBERT + rule-feature fusion trunk."""
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        transformer_hidden_size = int(self.transformer.config.hidden_size)

        self.rule_processor = nn.Sequential(
            nn.Linear(rule_feature_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, hidden_size // 4),
        )

        combined_size = transformer_hidden_size + hidden_size // 4
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rule_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return class logits for a tokenised batch.

        Args:
            input_ids: ``(batch, seq_len)`` token ids.
            attention_mask: ``(batch, seq_len)`` attention mask.
            rule_features: ``(batch, NUM_RULE_FEATURES)`` rule features.

        Returns:
            ``(batch, num_classes)`` logits.
        """
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = transformer_output.last_hidden_state

        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask_expanded, 1)
        summed_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_output = summed / summed_mask

        rule_output = self.rule_processor(rule_features)
        return self.classifier(torch.cat([pooled_output, rule_output], dim=1))
