from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class LineVulVerdict:
    verdict: str
    reason: str
    score: float
    checkpoint: str

    def as_json(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "score": self.score,
            "model_path": self.checkpoint,
            "constraint_result": None,
            "operations": [],
        }


class LineVulBaseline:
    """Function-level inference compatible with the official LineVul RQ1 model."""

    def __init__(
        self,
        codebert_path: str | Path,
        checkpoint_path: str | Path,
        *,
        threshold: float = 0.5,
        device: str = "auto",
        block_size: int = 512,
    ) -> None:
        try:
            import torch
            import torch.nn as nn
            from transformers import (
                RobertaConfig,
                RobertaForSequenceClassification,
                RobertaTokenizer,
            )
        except ImportError as error:
            raise RuntimeError(
                "LineVul baseline requires torch and transformers; install requirements.txt"
            ) from error

        self._torch = torch
        self.codebert_path = Path(codebert_path).expanduser()
        self.checkpoint = Path(checkpoint_path).expanduser()
        if not (self.codebert_path / "config.json").is_file():
            raise FileNotFoundError(
                f"local CodeBERT base model not found: {self.codebert_path}"
            )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                "LineVul checkpoint not found: "
                f"{self.checkpoint}. Download the official 12heads_linevul_model.bin."
            )

        config = RobertaConfig.from_pretrained(
            self.codebert_path,
            local_files_only=True,
        )
        # Match the official LineVul RQ1 construction.
        config.num_labels = 1
        config.num_attention_heads = 12

        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.codebert_path,
            local_files_only=True,
        )
        encoder = RobertaForSequenceClassification.from_pretrained(
            self.codebert_path,
            config=config,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )

        class RobertaClassificationHead(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
                self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
                self.out_proj = nn.Linear(cfg.hidden_size, 2)

            def forward(self, features):
                x = features[:, 0, :]
                x = self.dropout(x)
                x = self.dense(x)
                x = torch.tanh(x)
                x = self.dropout(x)
                return self.out_proj(x)

        class LineVulModel(nn.Module):
            def __init__(self, wrapped_encoder, cfg):
                super().__init__()
                self.encoder = wrapped_encoder
                self.classifier = RobertaClassificationHead(cfg)

            def forward(self, input_ids):
                hidden = self.encoder.roberta(
                    input_ids,
                    attention_mask=input_ids.ne(1),
                )[0]
                logits = self.classifier(hidden)
                return torch.softmax(logits, dim=-1)

        self.model = LineVulModel(encoder, config)
        state = torch.load(self.checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"unsupported LineVul checkpoint format: {self.checkpoint}")

        incompatible = self.model.load_state_dict(state, strict=False)
        required = {
            "classifier.dense.weight",
            "classifier.dense.bias",
            "classifier.out_proj.weight",
            "classifier.out_proj.bias",
        }
        missing_required = sorted(required & set(incompatible.missing_keys))
        if missing_required:
            raise ValueError(
                "LineVul checkpoint is missing trained classifier parameters: "
                + ", ".join(missing_required)
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.threshold = float(threshold)
        self.block_size = int(block_size)
        self.signature = self._signature()

    def _signature(self) -> str:
        stat = self.checkpoint.stat()
        payload = {
            "baseline": "LineVul",
            "codebert_path": str(self.codebert_path.resolve()),
            "checkpoint": str(self.checkpoint.resolve()),
            "checkpoint_size": stat.st_size,
            "checkpoint_mtime_ns": stat.st_mtime_ns,
            "threshold": self.threshold,
            "block_size": self.block_size,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    def _input_ids(self, function_source: str):
        code_tokens = self.tokenizer.tokenize(str(function_source))[: self.block_size - 2]
        source_tokens = (
            [self.tokenizer.cls_token]
            + code_tokens
            + [self.tokenizer.sep_token]
        )
        source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens)
        source_ids += [self.tokenizer.pad_token_id] * (
            self.block_size - len(source_ids)
        )
        return self._torch.tensor([source_ids], dtype=self._torch.long, device=self.device)

    def predict(self, function_source: str) -> LineVulVerdict:
        input_ids = self._input_ids(function_source)
        with self._torch.inference_mode():
            probabilities = self.model(input_ids)[0]
        score = float(probabilities[1].item())
        vulnerable = score > self.threshold
        return LineVulVerdict(
            "VULNERABLE" if vulnerable else "NOT_DETECTED",
            f"LineVul vulnerable probability={score:.6f} threshold={self.threshold:.3f}",
            score,
            str(self.checkpoint),
        )
