# Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Any

import torch

from .logging import get_local_logger

logger = get_local_logger(__name__)


def mask_aware_batch_mean(
    tensor: torch.Tensor, valid_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Mean over spatial dims with per-sample valid-pixel masking.

    Args:
        tensor: [B, H, W] value map.
        valid_mask: [B, H, W] boolean mask (True = valid).

    Returns:
        Scalar tensor (mean across all batches).
    """
    B, H, W = tensor.shape
    tensor = tensor.clone()

    if valid_mask is not None:
        tensor[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))  # [B]

        if (n == 0).any():
            logger.warning("Some batches have no valid pixels. Counts: %s", n.tolist())
            n_safe = torch.clamp(n, min=1)
            result = torch.sum(tensor, (-1, -2)) / n_safe
            result = torch.where(
                n > 0,
                result,
                torch.tensor(float("nan"), dtype=result.dtype, device=result.device),
            )
            return torch.nanmean(result)
        return torch.sum(tensor, (-1, -2)) / n
    else:
        return torch.sum(tensor, (-1, -2)) / (H * W)


def abs_relative_difference(
    pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Absolute relative difference: |pred - gt| / gt.  Returns per-pixel map."""
    return torch.abs(pred - gt) / (gt + eps)


def squared_relative_difference(
    pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Squared relative difference: (pred - gt)^2 / gt.  Returns per-pixel map."""
    return torch.pow(pred - gt, 2) / (gt + eps)


def absrel(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Mean Absolute Relative error (AbsRel).

    Args:
        pred: [B, H, W] or [H, W] predicted depth.
        gt:   [B, H, W] or [H, W] ground-truth depth.
        valid_mask: optional boolean mask.

    Returns:
        Scalar tensor.
    """
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)
    err = abs_relative_difference(pred, gt, eps=eps)
    return mask_aware_batch_mean(err, valid_mask).mean()


def sqrel(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean Squared Relative error (SqRel)."""
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)
    err = squared_relative_difference(pred, gt, eps=eps)
    return mask_aware_batch_mean(err, valid_mask).mean()


def rmse(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Root Mean Squared Error (RMSE)."""
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)
    diff = pred - gt
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = pred.shape[-1] * pred.shape[-2]
    mse = torch.sum(diff.pow(2), (-1, -2)) / n
    return torch.sqrt(mse).mean()


def rmse_log(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Root Mean Squared Log Error."""
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)
    diff = torch.log(pred) - torch.log(gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = pred.shape[-1] * pred.shape[-2]
    mse = torch.sum(diff.pow(2), (-1, -2)) / n
    return torch.sqrt(mse).mean()


def log10_error(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean absolute log10 error."""
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)
    if valid_mask is not None:
        diff = torch.abs(torch.log10(pred[valid_mask]) - torch.log10(gt[valid_mask]))
    else:
        diff = torch.abs(torch.log10(pred) - torch.log10(gt))
    return diff.mean()


def silog(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale-Invariant Logarithmic error (SIlog, in %)."""
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)
    diff = torch.log(pred) - torch.log(gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = gt.shape[-2] * gt.shape[-1]
    first_term = torch.sum(diff.pow(2), (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    return torch.sqrt(torch.mean(first_term - second_term)) * 100


def irmse(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inverse RMSE (iRMSE): RMSE computed on inverse depth."""
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)
    diff = (1.0 / pred) - (1.0 / gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = pred.shape[-1] * pred.shape[-2]
    mse = torch.sum(diff.pow(2), (-1, -2)) / n
    return torch.sqrt(mse).mean()


def _threshold_acc(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fraction of pixels where max(pred/gt, gt/pred) < threshold."""
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)
    d = torch.max(pred / gt, gt / pred)
    bit = (d < threshold).float()
    if valid_mask is not None:
        bit[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = pred.shape[-1] * pred.shape[-2]
    return (torch.sum(bit, (-1, -2)) / n).mean()


def delta1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Delta1 accuracy (threshold = 1.25)."""
    return _threshold_acc(pred, gt, 1.25, valid_mask)


def delta2(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Delta2 accuracy (threshold = 1.25^2)."""
    return _threshold_acc(pred, gt, 1.25**2, valid_mask)


def delta3(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Delta3 accuracy (threshold = 1.25^3)."""
    return _threshold_acc(pred, gt, 1.25**3, valid_mask)


def compute_depth_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Compute a standard set of depth metrics.

    Args:
        pred: [B, H, W] or [H, W] predicted depth (metric scale, > 0).
        gt:   [B, H, W] or [H, W] ground-truth depth (> 0 where valid).
        valid_mask: optional boolean mask (True = valid).  If *None* and gt
                    contains zeros, pixels with ``gt <= 0`` are auto-masked.

    Returns:
        Dictionary with keys: absrel, sqrel, rmse, rmse_log, log10,
        silog, irmse, delta1, delta2, delta3.
    """
    pred, gt, valid_mask = _ensure_batched(pred, gt, valid_mask)

    # Auto-mask: gt must be positive for all ratio-based metrics
    if valid_mask is None:
        valid_mask = gt > 0
    else:
        valid_mask = valid_mask & (gt > 0)

    # Clamp pred to positive for log-based metrics
    pred = pred.clamp(min=1e-6)

    return {
        "absrel": absrel(pred, gt, valid_mask).item(),
        "sqrel": sqrel(pred, gt, valid_mask).item(),
        "rmse": rmse(pred, gt, valid_mask).item(),
        "rmse_log": rmse_log(pred, gt, valid_mask).item(),
        "log10": log10_error(pred, gt, valid_mask).item(),
        "silog": silog(pred, gt, valid_mask).item(),
        "irmse": irmse(pred, gt, valid_mask).item(),
        "delta1": delta1(pred, gt, valid_mask).item(),
        "delta2": delta2(pred, gt, valid_mask).item(),
        "delta3": delta3(pred, gt, valid_mask).item(),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    """Return a single-line human-readable summary string."""
    parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
    return " | ".join(parts)


def average_metrics(
    list_of_dicts: list[dict[str, Any]],
    ignore_keys: list[str] | None = None,
) -> dict[str, Any]:
    """
    Average numeric values across a list of metric dicts.

    Non-numeric or NaN values are skipped per key.
    """
    keys_to_ignore = set(ignore_keys) if ignore_keys else set()
    all_keys = sorted({k for d in list_of_dicts for k in d.keys()} - keys_to_ignore)

    result: dict[str, Any] = {}
    for k in all_keys:
        values = [
            d[k]
            for d in list_of_dicts
            if isinstance(d.get(k), (int, float)) and d.get(k) == d.get(k)  # NaN check
        ]
        if not values:
            logger.warning("No valid values found for key `%s`", k)
            result[k] = float("nan")
        else:
            result[k] = sum(values) / len(values)

    return result


def _ensure_batched(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Ensure tensors have a batch dimension [B, H, W]."""
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    if gt.ndim == 2:
        gt = gt.unsqueeze(0)
    if valid_mask is not None and valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(0)
    return pred, gt, valid_mask
