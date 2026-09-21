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
    output = [f"\n{title}", "┌─" + border.replace("┼", "┬") + "─┐", line(columns),
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
        if row["event"] == "step":
            tqdm.write(f"  Epoch {row['epoch']} │ step {row['step']:>5} │ "
                       f"window loss {row['loss']:.5f} │ lr {row['learning_rate']:.2e} │ "
                       f"samples {row['samples_seen']}", file=sys.stdout)
        else:
            validation = row.get("validation") or {}
            print_table(f"Epoch {row['epoch']} complete", ["Train loss", "Steps", "Threshold", "MCC", "AUC", "F1"], [[
                f"{row['train_loss']:.5f}", row["optimizer_steps"],
                f"{row['validation_threshold']:.2f}" if validation else "fixed epochs",
                *[f"{validation[k]:.4f}" if validation.get(k) is not None else "—" for k in ("mcc", "auc", "f1")]]])
