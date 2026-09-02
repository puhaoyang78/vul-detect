from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeBERTVerdict:
    verdict: str
    reason: str
    score: float
    model_path: str

    def as_json(self) -> dict[str, object]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "score": self.score,
            "model_path": self.model_path,
            "constraint_result": None,
            "operations": [],
        }


def _resolve_checkpoint(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if (root / "config.json").is_file():
        return root
    if not root.is_dir():
        raise FileNotFoundError(f"CodeBERT checkpoint directory not found: {root}")

    candidates = []
    for config in root.rglob("config.json"):
        parent = config.parent
        if "codebert" in str(parent).lower() or "checkpoint" in parent.name.lower():
            candidates.append(parent)
    if not candidates:
        raise FileNotFoundError(
            f"no local HuggingFace CodeBERT checkpoint with config.json found under {root}"
        )
    candidates.sort(key=lambda item: (len(item.parts), str(item)))
    return candidates[0]


class CodeBERTBaseline:
    """Local function-level CodeBERT sequence-classification baseline."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        threshold: float = 0.5,
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "CodeBERT baseline requires torch and transformers; install requirements.txt"
            ) from error

        self._torch = torch
        self.checkpoint = _resolve_checkpoint(model_path)
        config = AutoConfig.from_pretrained(self.checkpoint, local_files_only=True)
        architectures = tuple(config.architectures or ())
        if int(getattr(config, "num_labels", 0) or 0) < 2:
            raise ValueError(
                f"CodeBERT checkpoint must have at least two labels: {self.checkpoint}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, local_files_only=True, use_fast=True
        )
        self.model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint,
            local_files_only=True,
            output_loading_info=True,
        )
        missing_head = [
            key
            for key in loading_info.get("missing_keys", [])
            if any(token in key.lower() for token in ("classifier", "score", "out_proj"))
        ]
        if missing_head:
            raise ValueError(
                "the configured CodeBERT path does not contain a trained classification "
                f"head ({', '.join(missing_head)}). Point --codebert-path to a fine-tuned "
                "function-level vulnerability checkpoint rather than a base encoder."
            )
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.threshold = float(threshold)
        self.max_length = int(max_length)
        self.positive_index = self._positive_index(config)
        self.signature = self._signature(config)

    def _positive_index(self, config) -> int:
        id2label = {
            int(key): str(value).lower()
            for key, value in (getattr(config, "id2label", {}) or {}).items()
        }
        for index, label in id2label.items():
            if any(token in label for token in ("vulnerable", "vulnerability", "positive", "unsafe")):
                return index
        return 1

    def _signature(self, config) -> str:
        config_path = self.checkpoint / "config.json"
        payload = {
            "checkpoint": str(self.checkpoint.resolve()),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "architectures": list(config.architectures or ()),
            "num_labels": int(config.num_labels),
            "positive_index": self.positive_index,
            "threshold": self.threshold,
            "max_length": self.max_length,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    def predict(self, function_source: str) -> CodeBERTVerdict:
        encoded = self.tokenizer(
            function_source,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            logits = self.model(**encoded).logits[0]
            probabilities = self._torch.softmax(logits, dim=-1)
        score = float(probabilities[self.positive_index].item())
        vulnerable = score >= self.threshold
        return CodeBERTVerdict(
            "VULNERABLE" if vulnerable else "NOT_DETECTED",
            (
                f"CodeBERT vulnerable probability={score:.6f} "
                f"threshold={self.threshold:.3f}"
            ),
            score,
            str(self.checkpoint),
        )
