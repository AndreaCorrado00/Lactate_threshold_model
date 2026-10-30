import numpy as np
import pytest

from src.feature_selection import forward_select
from src.preprocessing import FoldPreprocessor


def test_forward_selects_signal_stops_on_noise_and_preserves_inner_isolation(monkeypatch):
    from sklearn.linear_model import LinearRegression
    rows = [dict(test_id=str(i), gender='F' if i % 2 else 'M', age_years=30,
                 mass_kg=float(i), height_m=1., P_TLac_W=float(100 + 2 * i)) for i in range(36)]
    fit_ids = []
    original_fit = FoldPreprocessor.fit
    def tracked_fit(prep, train):
        assert prep.scale is False
        fit_ids.append({r['test_id'] for r in train})
        return original_fit(prep, train)
    monkeypatch.setattr(FoldPreprocessor, 'fit', tracked_fit)
    selected, audit = forward_select(rows, ['height_m', 'mass_kg'],
                                    lambda: LinearRegression(), False, 42, 3, .1)
    assert selected == ['mass_kg']
    assert audit['stop_reason'] == 'miglioramento insufficiente'
    assert len(audit['candidates']) == 3
    assert len(fit_ids) == 9
    for assignment in range(3):
        valid = {r['test_id'] for r in audit['inner_folds'] if r['fold'] == assignment}
        assert all(not (ids & valid) for ids in fit_ids[assignment::3])
    assert max(map(len, fit_ids)) == 24


def test_scaling_disabled_preserves_units_and_imputes():
    rows = [dict(gender='F', mass_kg=50), dict(gender='F', mass_kg=70)]
    prep = FoldPreprocessor(['mass_kg'], scale=False).fit(rows)
    assert prep.scaler is None
    np.testing.assert_allclose(prep.transform([dict(gender='F', mass_kg=None)]), [[60]])


@pytest.mark.parametrize('last_score,expected', [(9.875, ['mass_kg', 'height_m']),
                                               (9.9375, ['mass_kg'])])
def test_forward_accepts_threshold_and_rejects_smaller_gain(monkeypatch, last_score, expected):
    import src.feature_selection as module
    from sklearn.linear_model import LinearRegression
    rows = [dict(test_id=str(i), gender='F' if i % 2 else 'M', age_years=30,
                 mass_kg=float(i), height_m=1., P_TLac_W=float(100 + i)) for i in range(12)]
    scores = iter([10.] * 3 + [11.] * 3 + [last_score] * 3)
    monkeypatch.setattr(module, 'mean_absolute_error', lambda y, pred: next(scores))
    selected, _ = forward_select(rows, ['mass_kg', 'height_m'],
                                 lambda: LinearRegression(), False, 42, 3, .125)
    assert selected == expected
