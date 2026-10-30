import json

import joblib
import pytest
from sklearn.linear_model import Ridge

from src.evaluate import evaluate
from src.preprocessing import FoldPreprocessor
from src.split_dataset import csv_bytes, digest


def test_requires_explicit_test_access_before_reading(tmp_path):
    with pytest.raises(ValueError, match='allow-test-access'):
        evaluate(tmp_path / 'missing', tmp_path / 'split', tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_evaluation_uses_frozen_preprocessing_and_checks_integrity(tmp_path, monkeypatch):
    run, split = tmp_path / 'run', tmp_path / 'split'
    run.mkdir()
    split.mkdir()
    rows = [{'test_id': str(i), 'gender': 'M' if i % 2 else 'F',
             'age_years': 30, 'mass_kg': 50 + i, 'P_TLac_W': 100 + i * 2}
            for i in range(8)]
    train, test = rows[:4], rows[4:]
    prep = FoldPreprocessor(['mass_kg'], scale=True).fit(train)
    estimator = Ridge().fit(prep.transform(train), [r['P_TLac_W'] for r in train])
    train_content = csv_bytes(train, list(train[0]))
    test_content = csv_bytes(test, list(test[0]))
    parent_hash = digest(train_content)
    (split / 'train.csv').write_bytes(train_content)
    (split / 'test.csv').write_bytes(test_content)
    (split / 'split_manifest.json').write_text(json.dumps({
        'train_sha256': parent_hash, 'test_sha256': digest(test_content), 'n_test': 4}))
    (split / 'train_manifest.json').write_text(json.dumps({'train_sha256': parent_hash}))
    (run / 'config.json').write_text(json.dumps({'split_dir': str(split)}))
    (run / 'summary.json').write_text(json.dumps({'refit': True, 'train_sha256': parent_hash}))
    joblib.dump({'model': estimator, 'preprocessor': prep, 'features': ['mass_kg'],
                 'train_sha256': parent_hash}, run / 'model.joblib')
    def no_fit(*args, **kwargs):
        raise AssertionError('Fit vietato durante la valutazione')
    monkeypatch.setattr(FoldPreprocessor, 'fit', no_fit)
    monkeypatch.setattr(Ridge, 'fit', no_fit)
    result = evaluate(run, split, tmp_path / 'output', allow_test_access=True)
    assert result['n_test'] == 4
    assert (tmp_path / 'output' / 'validation.html').exists()
    with pytest.raises(FileExistsError):
        evaluate(run, split, tmp_path / 'output', allow_test_access=True)
    (split / 'test.csv').write_bytes(test_content + b'\n')
    with pytest.raises(ValueError, match='Hash'):
        evaluate(run, split, tmp_path / 'changed', allow_test_access=True)
