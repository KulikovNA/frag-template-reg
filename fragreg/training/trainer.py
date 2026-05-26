"""Training loop helpers."""

from __future__ import annotations

from itertools import islice
import time
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
    gradient_accumulation_steps: int = 1,
) -> dict[str, float]:
    model.train()
    device = torch.device(device)
    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
    totals: dict[str, float] = {}
    num_batches = 0
    optimizer_steps = 0
    epoch_start = time.perf_counter()

    iterable = dataloader if max_batches is None else islice(dataloader, max_batches)
    total = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    progress = tqdm(iterable, total=total, desc=f"train epoch {epoch}", leave=False)
    optimizer.zero_grad(set_to_none=True)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["points_C"])
        loss_dict = loss_fn(outputs, batch)
        loss = loss_dict["loss"]
        (loss / gradient_accumulation_steps).backward()

        num_batches += 1
        if num_batches % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        progress.set_postfix(loss=totals["loss"] / num_batches)

    if num_batches == 0:
        return {"loss": 0.0}
    remainder = num_batches % gradient_accumulation_steps
    if remainder != 0:
        scale = gradient_accumulation_steps / remainder
        if scale != 1.0:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    epoch_time_sec = time.perf_counter() - epoch_start
    metrics = {key: value / num_batches for key, value in totals.items()}
    metrics["epoch_time_sec"] = epoch_time_sec
    metrics["seconds_per_batch"] = epoch_time_sec / max(num_batches, 1)
    metrics["optimizer_steps"] = float(optimizer_steps)
    metrics["gradient_accumulation_steps"] = float(gradient_accumulation_steps)
    return metrics
