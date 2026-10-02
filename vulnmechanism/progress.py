"""Live terminal reporting and machine-readable training history."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tqdm.auto import tqdm


def print_table(title: str, columns: list[str], rows: list[list[object]]) -> None:
    values = [[str(value) for value in row] for row in rows]
    widths = [max(len(name), *(len(row[i]) for row in values)) if values else len(name)
              for i, name in enumerate(columns)]
    def line(row):
        return "│ " + " │ ".join(value.ljust(width) for value, width in zip(row, widths)) + " │"
    border = "─┼─".join("─" * width for width in widths)
    heading = f"\033[1;36m{title}\033[0m" if sys.stdout.isatty() else title
    output = [f"\n{heading}", "┌─" + border.replace("┼", "┬") + "─┐", line(columns),
              "├─" + border + "─┤", *[line(row) for row in values],
              "└─" + border.replace("┼", "┴") + "─┘"]
    tqdm.write("\n".join(output), file=sys.stdout)


class TrainingProgress:
    def __init__(self, checkpoint: str | Path):
        self.path = Path(checkpoint).with_suffix(".training.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def add(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        show_training_event(row)


def training_bar(iterable, **kwargs):
    """One live line on a terminal; no carriage-return spam in redirected logs."""
    options = dict(file=sys.stdout, dynamic_ncols=True, mininterval=1,
                   leave=False, disable=not sys.stdout.isatty(), colour="cyan", unit="batch",
                   bar_format="{desc}  {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} "
                              "[{elapsed}<{remaining}]{postfix}")
    options.update(kwargs)
    return tqdm(iterable, **options)


def show_training_event(row):
    """Presentation only: step events stay in the existing machine-readable history."""
    if row.get("event") != "epoch":
        return
    validation = row.get("validation") or {}
    value = lambda k: f"{validation[k]:.4f}" if validation.get(k) is not None else "—"
    print_table(f"{row.get('variant', 'Training')} · Epoch {row['epoch']} complete",
                ["Loss", "Steps", "Threshold", "MCC", "AUC", "F1"], [[
                    f"{row['train_loss']:.5f}", row['optimizer_steps'],
                    f"{row['validation_threshold']:.4f}" if validation else "fixed epochs",
                    value('mcc'), value('auc'), value('f1')]])
