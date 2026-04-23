import numpy as np

from laluzqueopaca.evaluation.metrics import iou_score


def test_iou_score_perfect_match():
    y_true = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    y_pred = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    assert iou_score(y_true, y_pred) == 1.0

