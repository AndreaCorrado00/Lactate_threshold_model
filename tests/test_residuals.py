import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.residuals.model import OffsetRegressor
from src.residuals.train import run_training, comparison_metrics
from src.split_dataset import create_split, csv_bytes


def synthetic_rows(n=60):
    return [{'test_id': f'R{i}', 'gender': 'M' if i % 2 else 'F', 'age_years': 30,
             'max_power_W': 180 + 10 * (i % 9), 'mass_kg': 55 + i % 11,
             'P_TLac_W': .8 * (180 + 10 * (i % 9)) + 3 * (i % 11 - 5)}
            for i in range(n)]


def test_k_and_preprocessing_fit_only_training_and_reconstruction():
    train = synthetic_rows(30)
    model = OffsetRegressor(['mass_kg'], 'ridge', {'alpha': 1.}, scaling=True).fit(train)
    x = np.array([r['max_power_W'] for r in train])
    y = np.array([r['P_TLac_W'] for r in train])
    assert model.k_ == pytest.approx(x @ y / (x @ x))
    valid = [{'gender': 'F', 'mass_kg': None, 'max_power_W': 900}]
    mean = model.preprocessor_.scaler.mean_.copy()
    baseline, delta, prediction = model.predict_parts(valid)
    np.testing.assert_allclose(prediction, baseline + delta)
    np.testing.assert_allclose(baseline, [model.k_ * 900])
    np.testing.assert_array_equal(mean, model.preprocessor_.scaler.mean_)
    changed = [{**valid[0], 'P_TLac_W': 99999}]
    np.testing.assert_allclose(model.predict(changed), prediction)
    with pytest.raises(ValueError):
        model.predict([{**valid[0], 'max_power_W': None}])


@pytest.mark.parametrize('name,params', [('ridge', {'alpha': 1}),
    ('random_forest', {'n_estimators': 3, 'max_depth': 2}),
    ('xgboost', {'n_estimators': 3, 'max_depth': 2}),
    ('bagging', {'n_estimators': 3})])
def test_other_models_roundtrip(name, params, tmp_path):
    rows = synthetic_rows(30)
    model = OffsetRegressor(['mass_kg'], name, params, scaling=name == 'ridge').fit(rows)
    joblib.dump(model, tmp_path / 'model.joblib')
    np.testing.assert_allclose(joblib.load(tmp_path / 'model.joblib').predict(rows), model.predict(rows))


def test_direct_control_and_fixed_offset_are_explicit():
    rows = synthetic_rows(30)
    model = OffsetRegressor(['mass_kg'], 'ridge', {'alpha': 1}, target_mode='direct',
                            proportional={'mode': 'fixed', 'k': .7986}).fit(rows)
    assert model.k_ == .7986
    np.testing.assert_allclose(model.predict(rows), model.estimator_.predict(model.preprocessor_.transform(rows)))


def test_metrics_zero_delta_and_gain():
    result = comparison_metrics(np.array([80., 160.]), np.array([80., 160.]), np.array([80., 160.]))
    assert result['q_delta'] is None
    assert result['delta_correlation'] is None
    result = comparison_metrics(np.array([90., 150.]), np.array([80., 160.]), np.array([90., 150.]))
    assert result['q_delta'] == pytest.approx(1)
    assert result['mae_gain_W'] == pytest.approx(10)


@pytest.mark.parametrize('refit', [False, True])
def test_nested_cv_isolation_baseline_and_refit(tmp_path, monkeypatch, refit):
    rows = synthetic_rows()
    source = tmp_path / 'source.csv'
    source.write_bytes(csv_bytes(rows, list(rows[0])))
    split = tmp_path / 'split'
    create_split({'source': str(source), 'output': str(split), 'cv_folds': 3, 'seed': 42})
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path.name not in ('test.csv', 'split_manifest.json')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    sizes = []
    original_fit = OffsetRegressor.fit
    def tracked_fit(self, fit_rows):
        sizes.append(len(fit_rows))
        return original_fit(self, fit_rows)
    monkeypatch.setattr(OffsetRegressor, 'fit', tracked_fit)
    config = {'split_dir': str(split), 'features': ['mass_kg', 'gender'], 'seed': 42,
              'target_mode': 'residual', 'proportional': {'mode': 'fit'}, 'scaling': True,
              'importance_repeats': 2, 'models': {'ridge': {'alpha': 1}},
              'tuning': {'method': 'random', 'n_iter': 2, 'inner_folds': 2,
                         'param_spaces': {'ridge': {'alpha': [1, 10]}}}}
    output = tmp_path / 'run'
    result = run_training(config, 'ridge', output, refit=refit)
    assert result['n_oof'] == 48 and result['test_accessed'] is False
    assert {16, 32} <= set(sizes)
    assert (48 in sizes) == refit
    predictions = list(csv.DictReader((output / 'predictions_oof.csv').open()))
    assert len({r['test_id'] for r in predictions}) == 48
    for r in predictions:
        assert float(r['predicted_W']) == pytest.approx(float(r['baseline_W']) + float(r['predicted_delta_W']))
    audit = json.loads((output / 'folds.json').read_text())
    by_id = {r['test_id']: r for r in rows}
    for fold in audit:
        outer_valid = set(fold['validation_ids'])
        assert not set(fold['train_ids']) & outer_valid
        for part in fold['tuning']['partitions']:
            assert not outer_valid & set(part['train_ids'] + part['validation_ids'])
            assert not set(part['train_ids']) & set(part['validation_ids'])
            xx = np.array([by_id[i]['max_power_W'] for i in part['train_ids']])
            yy = np.array([by_id[i]['P_TLac_W'] for i in part['train_ids']])
            assert part['k'] == pytest.approx(xx @ yy / (xx @ xx))
    assert (output / 'model.joblib').exists() == refit
    if refit:
        predictor = joblib.load(output / 'model.joblib')['predictor']
        assert np.isfinite(predictor.predict(rows)).all()


def test_rejects_max_power_as_correction_feature_before_reading(tmp_path):
    with pytest.raises(ValueError, match='feature'):
        run_training({'features': ['max_power_W'], 'split_dir': 'absent'}, 'ridge', tmp_path / 'out')
    assert not (tmp_path / 'out').exists()
