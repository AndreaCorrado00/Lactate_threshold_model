import numpy as np
import pytest

from src.proportional_baseline import proportional_oof


def test_validation_targets_do_not_affect_own_predictions():
    x = np.array([100., 200., 100., 200.])
    y = np.array([70., 140., 90., 180.])
    folds = np.array([0, 0, 1, 1])
    pred, coefficients = proportional_oof(x, y, folds)
    np.testing.assert_allclose(pred, [90, 180, 70, 140])
    assert coefficients == pytest.approx({0: .9, 1: .7})
    changed = y.copy()
    changed[:2] *= 2
    new_pred, _ = proportional_oof(x, changed, folds)
    np.testing.assert_allclose(new_pred[:2], pred[:2])


def test_rejects_invalid_maximum():
    with pytest.raises(ValueError):
        proportional_oof([0, 100], [70, 80], [0, 1])
