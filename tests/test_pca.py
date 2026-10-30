import joblib
import numpy as np
import pytest

from src.preprocessing import FoldPreprocessor


def test_pca_is_frozen_on_validation_and_gender_is_passthrough(tmp_path):
    rows = [{'gender': 'F' if i % 2 else 'M', 'mass_kg': 50+i,
             'height_m': 1.5+i/100} for i in range(12)]
    prep = FoldPreprocessor(['mass_kg', 'gender', 'height_m'], scale=True,
                            pca={'n_components': .95}).fit(rows)
    mean = prep.scaler.mean_.copy()
    components = prep.pca_.components_.copy()
    valid = [{'gender': 'F', 'mass_kg': None, 'height_m': 9.}]
    transformed = prep.transform(valid)
    assert transformed.shape == (1, 2)
    assert transformed[0, -1] == 0
    np.testing.assert_array_equal(prep.scaler.mean_, mean)
    np.testing.assert_array_equal(prep.pca_.components_, components)
    joblib.dump(prep, tmp_path / 'prep.joblib')
    np.testing.assert_allclose(joblib.load(tmp_path / 'prep.joblib').transform(valid), transformed)


def test_pca_requires_scaling_and_valid_dimension():
    with pytest.raises(ValueError):
        FoldPreprocessor(['mass_kg'], pca={'n_components': .95})
    with pytest.raises(ValueError):
        FoldPreprocessor(['mass_kg'], scale=True, pca={'n_components': True})


def test_legacy_preprocessor_without_pca_attribute_still_transforms():
    prep = FoldPreprocessor(['mass_kg']).fit([{'gender': 'F', 'mass_kg': 60}])
    prep.__dict__.pop('pca_config', None)
    np.testing.assert_allclose(prep.transform([{'gender': 'F', 'mass_kg': None}]), [[60]])


def test_nested_training_refit_and_exploration_never_read_holdout(tmp_path, monkeypatch):
    from pathlib import Path
    import json
    from src.split_dataset import csv_bytes, create_split
    from src.train import run_training, load_training
    from src.pca_report import build_pca_report
    rows = [{'test_id': str(i), 'gender': 'M' if i % 2 else 'F', 'age_years': 30,
             'mass_kg': 50+i/2, 'height_m': 1.5+(i % 7)/10, 'P_TLac_W': 100+2*i}
            for i in range(40)]
    source = tmp_path / 'source.csv'
    source.write_bytes(csv_bytes(rows, list(rows[0])))
    split = tmp_path / 'split'
    create_split({'source': str(source), 'output': str(split), 'cv_folds': 2})
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert path.name not in ('test.csv', 'split_manifest.json')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    from sklearn.decomposition import PCA
    original_fit = PCA.fit
    fit_sizes = []
    def traced_fit(self, x, *args, **kwargs):
        fit_sizes.append(len(x))
        return original_fit(self, x, *args, **kwargs)
    monkeypatch.setattr(PCA, 'fit', traced_fit)
    config = {'split_dir': str(split), 'features': ['mass_kg', 'height_m', 'gender'],
              'scaling': True, 'pca': {'n_components': .95}, 'importance_repeats': 2,
              'models': {'ridge': {'alpha': 1}}, 'tuning': {'method': 'random',
              'n_iter': 2, 'inner_folds': 2, 'param_spaces': {'ridge': {'alpha': [1, 10]}}}}
    result = run_training(config, 'ridge', tmp_path / 'run', refit=True)
    assert {8, 16, 32} <= set(fit_sizes)
    assert result['refit_pca']['n_components'] >= 1
    bundle = joblib.load(tmp_path / 'run/model.joblib')
    train, _, _, _ = load_training(split)
    assert np.isfinite(bundle['model'].predict(bundle['preprocessor'].transform(train))).all()
    build_pca_report(config, tmp_path / 'exploration')
    assert (tmp_path / 'exploration/pca.html').exists()
    meta = json.loads((tmp_path / 'exploration/summary.json').read_text())
    assert not meta['test_accessed']
