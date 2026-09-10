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


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float
    auc: float | None

    def as_json(self) -> dict[str, float | None]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'mcc': self.mcc,
            'auc': self.auc,
        }


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'invalid JSON at {path}:{line_number}: {error}') from error
            if not isinstance(record, dict):
                raise ValueError(f'{path}:{line_number}: each JSONL row must be an object')
            records.append(record)
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
    return str(record['raw_source'])


def _cpg_text(record: dict[str, object]) -> str:
    return (
        'TASK: classify whether this C/C++ function is vulnerable.\n'
        'FUNCTION:\n' + str(record['raw_source']) + '\n'
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
            outputs.append(encoded[row, lengths].float().cpu())
        return torch.cat(outputs, dim=0) if outputs else torch.empty((0, self.hidden_size))


class BinaryHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings).squeeze(-1)


def _labels(records: list[dict[str, object]]) -> torch.Tensor:
    labels = []
    for record in records:
        value = int(record['label'])
        if value not in {0, 1}:
            raise ValueError(f"{record.get('sample_key')}: label must be 0 or 1")
        labels.append(float(value))
    return torch.tensor(labels, dtype=torch.float32)


def _train_head(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    hidden_size: int,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> BinaryHead:
    model = BinaryHead(hidden_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(embeddings, labels), batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch_embeddings, batch_labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_embeddings), batch_labels)
            loss.backward()
            optimizer.step()
    return model.eval()


def _binary_auc(truth: torch.Tensor, scores: torch.Tensor) -> float | None:
    positives = int(truth.sum().item())
    negatives = int((~truth).sum().item())
    if positives == 0 or negatives == 0:
        return None

    pairs = sorted(
        ((float(score), bool(label)) for score, label in zip(scores.tolist(), truth.tolist())),
        key=lambda item: item[0],
    )
    rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (rank + (rank + end - index - 1)) / 2.0
        rank_sum += average_rank * sum(1 for _, label in pairs[index:end] if label)
        rank += end - index
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _classification_metrics(records: list[dict[str, object]], probabilities: torch.Tensor) -> Metrics:
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

    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    auc = _binary_auc(truth, probabilities)
    return Metrics(accuracy, precision, recall, f1, mcc, auc)


def train_models(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str,
    max_length: int = 2048,
    encoder_batch_size: int = 2,
    head_batch_size: int = 64,
    epochs: int = 20,
    learning_rate: float = 1e-3,
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

    labels = _labels(train_records)
    encoder = QwenEncoder(model_path, max_length=max_length, device=device)

    baseline_train = encoder.encode(
        [_baseline_text(record) for record in train_records], encoder_batch_size
    )
    cpg_train = encoder.encode(
        [_cpg_text(record) for record in train_records], encoder_batch_size
    )
    baseline = _train_head(
        baseline_train,
        labels,
        encoder.hidden_size,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=head_batch_size,
    )
    cpg = _train_head(
        cpg_train,
        labels,
        encoder.hidden_size,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=head_batch_size,
    )

    validation: dict[str, object] = {}
    if valid_records:
        baseline_valid = encoder.encode(
            [_baseline_text(record) for record in valid_records], encoder_batch_size
        )
        cpg_valid = encoder.encode(
            [_cpg_text(record) for record in valid_records], encoder_batch_size
        )
        with torch.no_grad():
            baseline_prob = torch.sigmoid(baseline(baseline_valid))
            cpg_prob = torch.sigmoid(cpg(cpg_valid))
        validation = {
            'baseline': _classification_metrics(valid_records, baseline_prob).as_json(),
            'cpg': _classification_metrics(valid_records, cpg_prob).as_json(),
        }

    checkpoint = {
        'model_path': model_path,
        'hidden_size': encoder.hidden_size,
        'max_length': max_length,
        'baseline_state': baseline.state_dict(),
        'cpg_state': cpg.state_dict(),
        'validation': validation,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, target)
    print(json.dumps({'checkpoint': str(target), 'validation': validation}, ensure_ascii=False))
    return checkpoint


def evaluate_models(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    *,
    split: str = 'test',
    encoder_batch_size: int = 2,
    device: str = 'auto',
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    records = [record for record in _read_jsonl(dataset_path) if _record_split(record) == split]
    if not records:
        raise ValueError(f'{split} split is empty')

    encoder = QwenEncoder(
        str(checkpoint['model_path']),
        max_length=int(checkpoint['max_length']),
        device=device,
    )
    baseline = BinaryHead(int(checkpoint['hidden_size']))
    baseline.load_state_dict(checkpoint['baseline_state'])
    baseline.eval()
    cpg = BinaryHead(int(checkpoint['hidden_size']))
    cpg.load_state_dict(checkpoint['cpg_state'])
    cpg.eval()

    baseline_embeddings = encoder.encode(
        [_baseline_text(record) for record in records], encoder_batch_size
    )
    cpg_embeddings = encoder.encode(
        [_cpg_text(record) for record in records], encoder_batch_size
    )
    with torch.no_grad():
        baseline_probability = torch.sigmoid(baseline(baseline_embeddings))
        cpg_probability = torch.sigmoid(cpg(cpg_embeddings))

    result = {
        'split': split,
        'samples': len(records),
        'positive': sum(int(record['label']) == 1 for record in records),
        'negative': sum(int(record['label']) == 0 for record in records),
        'baseline': _classification_metrics(records, baseline_probability).as_json(),
        'cpg': _classification_metrics(records, cpg_probability).as_json(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
