import numpy as np


def iou_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Compute intersection over union for binary masks."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return float((intersection + eps) / (union + eps))

