from typing import Iterable

import torch
from torchmetrics import Metric
from tqdm.auto import tqdm


def compute_metrics(model, loader, device, metrics: Metric | Iterable[Metric], show_progress=False):
    if isinstance(metrics, Metric):
        metrics = [metrics]
    model.to(device)
    metrics = [metric.to(device) for metric in metrics]
    for metric in metrics:
        metric.reset()
    model.eval()
    with torch.no_grad():
        for minibatch in tqdm(loader, disable=not show_progress):
            batch_x = {k: v.to(device) for k, v in minibatch.items() if k != "targets"}
            batch_y = minibatch["targets"].to(device)
            pred = model(batch_x)
            for metric in metrics:
                metric(pred, batch_y)
    res = []
    for metric in metrics:
        res.append(metric.compute())
        metric.reset()
    return res
