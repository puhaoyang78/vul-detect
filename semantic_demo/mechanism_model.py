from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from .mechanism import MECHANISM_COMPONENTS


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    pair_accuracy: float
    mechanism_accuracy: float | None = None

    def as_json(self) -> dict[str, float | None]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'pair_accuracy': self.pair_accuracy,
            'mechanism_accuracy': self.mechanism_accuracy,
        }


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f'invalid JSON at {path}:{line_number}: {error}') from error
    return records


def _split_for_key(key: str) -> str:
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return 'train'
    if bucket < 85:
        return 'valid'
    return 'test'


def _record_split(record: dict[str, object]) -> str:
    split = str(record.get('split') or '').lower()
    if split in {'train', 'valid', 'validation', 'test'}:
        return 'valid' if split == 'validation' else split
    return _split_for_key(str(record['sample_key']))


def _baseline_text(record: dict[str, object]) -> str:
    return str(record.get('raw_source') or record['canonical_source'])


def _proposed_text(record: dict[str, object], *, renamed: bool = False) -> str:
    source_key = 'renamed_canonical_source' if renamed else 'canonical_source'
    return (
        'TASK: infer security mechanism state from one C/C++ function.\n'
        'FUNCTION:\n' + str(record[source_key]) + '\n'
        'INTRA_FUNCTION_CPG:\n' + str(record['graph'])
    )


class QwenEncoder:
    def __init__(self, model_path: str, max_length: int = 2048, device: str = 'auto') -> None:
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.hidden_size = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 2) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = self.tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            encoded = self.model(**tokens).last_hidden_state
            lengths = tokens['attention_mask'].sum(dim=1) - 1
            row = torch.arange(encoded.size(0), device=self.device)
            pooled = encoded[row, lengths].float().cpu()
            outputs.append(pooled)
        return torch.cat(outputs, dim=0) if outputs else torch.empty((0, self.hidden_size))


class BaselineHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings).squeeze(-1)


class MechanismBottleneckHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.mechanism = nn.Linear(hidden_size, len(MECHANISM_COMPONENTS))
        self.classifier = nn.Linear(len(MECHANISM_COMPONENTS), 1)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mechanism_logits = self.mechanism(embeddings)
        vulnerability_logits = self.classifier(torch.sigmoid(mechanism_logits)).squeeze(-1)
        return mechanism_logits, vulnerability_logits


def _tensor_rows(records: list[dict[str, object]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.tensor([float(record['label']) for record in records], dtype=torch.float32)
    targets = torch.tensor([record['mechanism_targets'] for record in records], dtype=torch.float32)
    masks = torch.tensor([record['mechanism_mask'] for record in records], dtype=torch.float32)
    return labels, targets, masks


def _train_baseline(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    hidden_size: int,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> BaselineHead:
    model = BaselineHead(hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(train_embeddings, train_labels), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for embeddings, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(embeddings), labels)
            loss.backward()
            optimizer.step()
    return model.eval()


def _train_proposed(
    train_embeddings: torch.Tensor,
    labels: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    hidden_size: int,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    mechanism_weight: float,
) -> MechanismBottleneckHead:
    model = MechanismBottleneckHead(hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    vuln_loss_fn = nn.BCEWithLogitsLoss()
    mech_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    loader = DataLoader(
        TensorDataset(train_embeddings, labels, targets, masks),
        batch_size=batch_size,
        shuffle=True,
    )
    model.train()
    for _ in range(epochs):
        for embeddings, batch_labels, batch_targets, batch_masks in loader:
            optimizer.zero_grad()
            mechanism_logits, vulnerability_logits = model(embeddings)
            vuln_loss = vuln_loss_fn(vulnerability_logits, batch_labels)
            per_component = mech_loss_fn(mechanism_logits, batch_targets) * batch_masks
            denominator = batch_masks.sum().clamp_min(1.0)
            mechanism_loss = per_component.sum() / denominator
            loss = vuln_loss + mechanism_weight * mechanism_loss
            loss.backward()
            optimizer.step()
    return model.eval()


def _classification_metrics(
    records: list[dict[str, object]],
    probabilities: torch.Tensor,
    mechanism_logits: torch.Tensor | None = None,
) -> Metrics:
    predictions = probabilities >= 0.5
    truth = torch.tensor([int(record['label']) for record in records], dtype=torch.bool)
    tp = int((predictions & truth).sum())
    fp = int((predictions & ~truth).sum())
    tn = int((~predictions & ~truth).sum())
    fn = int((~predictions & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(records) if records else 0.0

    by_pair: dict[str, dict[str, bool]] = {}
    for record, prediction in zip(records, predictions.tolist()):
        by_pair.setdefault(str(record['sample_key']), {})[str(record['side'])] = bool(prediction)
    complete = [pair for pair in by_pair.values() if {'vulnerable', 'fixed'} <= set(pair)]
    pair_accuracy = (
        sum(pair['vulnerable'] and not pair['fixed'] for pair in complete) / len(complete)
        if complete else 0.0
    )

    mechanism_accuracy = None
    if mechanism_logits is not None and records:
        predicted = mechanism_logits >= 0
        targets = torch.tensor([record['mechanism_targets'] for record in records], dtype=torch.bool)
        masks = torch.tensor([record['mechanism_mask'] for record in records], dtype=torch.bool)
        denominator = int(masks.sum())
        if denominator:
            mechanism_accuracy = float(((predicted == targets) & masks).sum()) / denominator
    return Metrics(accuracy, precision, recall, f1, pair_accuracy, mechanism_accuracy)


def _rename_stability(original: torch.Tensor, renamed: torch.Tensor) -> dict[str, float]:
    if original.numel() == 0:
        return {'prediction_agreement': 0.0, 'mean_abs_probability_change': 0.0}
    agreement = float(((original >= 0.5) == (renamed >= 0.5)).float().mean())
    delta = float((original - renamed).abs().mean())
    return {'prediction_agreement': agreement, 'mean_abs_probability_change': delta}


def train_mechanism_models(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str,
    max_length: int = 2048,
    encoder_batch_size: int = 2,
    head_batch_size: int = 64,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    mechanism_weight: float = 1.0,
    seed: int = 42,
    device: str = 'auto',
) -> dict[str, object]:
    random.seed(seed)
    torch.manual_seed(seed)
    records = _read_jsonl(dataset_path)
    train_records = [record for record in records if _record_split(record) == 'train']
    valid_records = [record for record in records if _record_split(record) == 'valid']
    if not train_records:
        raise ValueError('training split is empty')

    encoder = QwenEncoder(model_path, max_length=max_length, device=device)
    baseline_train = encoder.encode([_baseline_text(record) for record in train_records], encoder_batch_size)
    proposed_train = encoder.encode([_proposed_text(record) for record in train_records], encoder_batch_size)
    labels, targets, masks = _tensor_rows(train_records)

    baseline = _train_baseline(
        baseline_train,
        labels,
        encoder.hidden_size,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=head_batch_size,
    )
    proposed = _train_proposed(
        proposed_train,
        labels,
        targets,
        masks,
        encoder.hidden_size,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=head_batch_size,
        mechanism_weight=mechanism_weight,
    )

    validation: dict[str, object] = {}
    if valid_records:
        baseline_valid = encoder.encode([_baseline_text(record) for record in valid_records], encoder_batch_size)
        proposed_valid = encoder.encode([_proposed_text(record) for record in valid_records], encoder_batch_size)
        with torch.no_grad():
            baseline_prob = torch.sigmoid(baseline(baseline_valid))
            mechanism_logits, proposed_logits = proposed(proposed_valid)
            proposed_prob = torch.sigmoid(proposed_logits)
        validation = {
            'baseline': _classification_metrics(valid_records, baseline_prob).as_json(),
            'proposed': _classification_metrics(valid_records, proposed_prob, mechanism_logits).as_json(),
        }

    checkpoint = {
        'model_path': model_path,
        'hidden_size': encoder.hidden_size,
        'max_length': max_length,
        'components': MECHANISM_COMPONENTS,
        'baseline_state': baseline.state_dict(),
        'proposed_state': proposed.state_dict(),
        'validation': validation,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)
    print(json.dumps({'checkpoint': str(target), 'validation': validation}, ensure_ascii=False))
    return checkpoint


def evaluate_mechanism_models(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    *,
    split: str = 'test',
    encoder_batch_size: int = 2,
    device: str = 'auto',
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if tuple(checkpoint['components']) != MECHANISM_COMPONENTS:
        raise ValueError('checkpoint mechanism component schema does not match current code')
    records = [record for record in _read_jsonl(dataset_path) if _record_split(record) == split]
    if not records:
        raise ValueError(f'{split} split is empty')

    encoder = QwenEncoder(
        str(checkpoint['model_path']),
        max_length=int(checkpoint['max_length']),
        device=device,
    )
    baseline = BaselineHead(int(checkpoint['hidden_size']))
    baseline.load_state_dict(checkpoint['baseline_state'])
    baseline.eval()
    proposed = MechanismBottleneckHead(int(checkpoint['hidden_size']))
    proposed.load_state_dict(checkpoint['proposed_state'])
    proposed.eval()

    baseline_embeddings = encoder.encode([_baseline_text(record) for record in records], encoder_batch_size)
    proposed_embeddings = encoder.encode([_proposed_text(record) for record in records], encoder_batch_size)
    renamed_baseline_embeddings = encoder.encode([str(record['renamed_source']) for record in records], encoder_batch_size)
    renamed_proposed_embeddings = encoder.encode([_proposed_text(record, renamed=True) for record in records], encoder_batch_size)
    with torch.no_grad():
        baseline_probability = torch.sigmoid(baseline(baseline_embeddings))
        renamed_baseline_probability = torch.sigmoid(baseline(renamed_baseline_embeddings))
        mechanism_logits, proposed_logits = proposed(proposed_embeddings)
        proposed_probability = torch.sigmoid(proposed_logits)
        _renamed_mechanism, renamed_proposed_logits = proposed(renamed_proposed_embeddings)
        renamed_proposed_probability = torch.sigmoid(renamed_proposed_logits)
    result = {
        'split': split,
        'samples': len(records),
        'pairs': len({str(record['sample_key']) for record in records}),
        'baseline': {
            **_classification_metrics(records, baseline_probability).as_json(),
            'rename_stability': _rename_stability(baseline_probability, renamed_baseline_probability),
        },
        'proposed': {
            **_classification_metrics(records, proposed_probability, mechanism_logits).as_json(),
            'rename_stability': _rename_stability(proposed_probability, renamed_proposed_probability),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
