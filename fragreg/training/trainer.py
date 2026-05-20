"""Training loop helpers."""

from __future__ import annotations

from itertools import islice
from typing import Any

import torch
from tqdm import tqdm

from fragreg.evaluation.evaluator import move_batch_to_device


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    epoch: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    device = torch.device(device)
    totals: dict[str, float] = {}
    num_batches = 0

    iterable = dataloader if max_batches is None else islice(dataloader, max_batches)
    total = len(dataloader) if max_batches is None else max_batches
    progress = tqdm(iterable, total=total, desc=f"train epoch {epoch}", leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["points_C"])
        loss_dict = loss_fn(outputs, batch)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        num_batches += 1
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        progress.set_postfix(loss=totals["loss"] / num_batches)

    if num_batches == 0:
        return {"loss": 0.0}
    return {key: value / num_batches for key, value in totals.items()}
