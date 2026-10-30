import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.split_dataset import create_split
from src.train import FoldPreprocessor, run_training


def synthetic_dataset(path):
    rows = []
    for i in range(80):
        rows.append({'test_id': f'S{i}', 'gender': 'F' if i % 2 else 'M',
                     'age_years': '' if i < 4 else [20, 30, 40][i % 3],
                     'mass_kg': '' if i in (10, 11) else 50 + i / 2,
                     'P_TLac_W': 100 + i * 2})
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def test_split_reproducible_disjoint_stratified_and_locked(tmp_path):
    source = tmp_path / 'source.csv'
    synthetic_dataset(source)
    config = {'source': str(source), 'output': str(tmp_path / 'split'),
              'seed': 42, 'test_fraction': .2, 'cv_folds': 5}
    result = create_split(config)
    assert (result['n_train'], result['n_test']) == (64, 16)
    def read(folder, name):
        with (folder / name).open(encoding='utf-8') as stream:
            return list(csv.DictReader(stream))
    folder = tmp_path / 'split'
    train, test = read(folder, 'train.csv'), read(folder, 'test.csv')
    assert not ({r['test_id'] for r in train} & {r['test_id'] for r in test})
    assert len({r['test_id'] for r in train + test}) == 80
    folds = read(folder, 'cv_folds.csv')
    assert {r['test_id'] for r in folds} == {r['test_id'] for r in train}
    sizes = Counter(r['fold'] for r in folds)
    assert max(sizes.values()) - min(sizes.values()) <= 1
    for stratum in {r['stratum'] for r in folds}:
        counts = Counter(r['fold'] for r in folds if r['stratum'] == stratum)
        assert max(counts.get(str(i), 0) for i in range(5)) - min(counts.get(str(i), 0) for i in range(5)) <= 1
    with pytest.raises(FileExistsError):
        create_split(config)
    create_split({**config, 'output': str(tmp_path / 'second')})
    assert (folder / 'train.csv').read_bytes() == (tmp_path / 'second/train.csv').read_bytes()


def test_preprocessing_is_fitted_only_on_fold_train():
    train = [{'gender': g, 'mass_kg': x} for g, x in [('F', 50), ('F', 60), ('M', 70), ('M', 90)]]
    prep = FoldPreprocessor(['mass_kg', 'gender'], scale=True).fit(train)
    validation = [{'gender': 'F', 'mass_kg': None}, {'gender': 'M', 'mass_kg': 10000}]
    transformed = prep.transform(validation)
    assert prep.imputer.medians['F']['mass_kg'] == 55
    assert prep.imputer.medians['M']['mass_kg'] == 80
    np.testing.assert_allclose(prep.scaler.mean_, [67.5, .5])
    assert np.isfinite(transformed).all()
    assert validation[0]['mass_kg'] is None


@pytest.mark.parametrize('model', ['ridge', 'random_forest', 'xgboost'])
def test_training_never_opens_test_and_default_does_not_refit(tmp_path, monkeypatch, model):
    source = tmp_path / 'source.csv'
    synthetic_dataset(source)
    split_dir = tmp_path / 'split'
    create_split({'source': str(source), 'output': str(split_dir), 'seed': 42,
                  'test_fraction': .2, 'cv_folds': 5})
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.name in ('test.csv', 'split_manifest.json'):
            raise AssertionError('Il training ha aperto il test o il manifest holdout')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    fit_sizes = []
    original_fit = FoldPreprocessor.fit
    def track_fit(preprocessor, rows):
        fit_sizes.append(len(rows))
        return original_fit(preprocessor, rows)
    monkeypatch.setattr(FoldPreprocessor, 'fit', track_fit)
    config = {'split_dir': str(split_dir), 'features': ['mass_kg', 'gender'],
              'seed': 42, 'importance_repeats': 2,
              'models': {'ridge': {'alpha': 1}, 'random_forest': {'n_estimators': 5},
                         'xgboost': {'n_estimators': 5, 'max_depth': 2}}}
    output = tmp_path / model
    result = run_training(config, model, output)
    assert result['refit'] is False
    assert result['n_oof'] == 64
    assert len(fit_sizes) == 5
    assert max(fit_sizes) < 64
    assert not (output / 'model.joblib').exists()
    assert (output / 'validation.html').exists()
    assert all(np.isfinite(result['oof'][k]) for k in ('mae_W', 'rmse_W', 'r2'))
    assert len(list(csv.DictReader((output / 'importance_folds.csv').open()))) == 10


def test_rejects_target_feature_before_training(tmp_path):
    with pytest.raises(ValueError, match='feature'):
        run_training({'features': ['P_TLac_W']}, 'ridge', tmp_path / 'run')


def test_explicit_refit_and_modified_train_rejected(tmp_path):
    import joblib
    source = tmp_path / 'source.csv'
    synthetic_dataset(source)
    folder = tmp_path / 'split'
    create_split({'source': str(source), 'output': str(folder), 'seed': 42,
                  'test_fraction': .2, 'cv_folds': 5})
    config = {'split_dir': str(folder), 'features': ['mass_kg'], 'importance_repeats': 2}
    output = tmp_path / 'refit'
    result = run_training(config, 'ridge', output, refit=True)
    assert result['refit'] is True
    artifact = joblib.load(output / 'model.joblib')
    assert artifact['features'] == ['mass_kg']
    assert artifact['preprocessor'].scaler.n_samples_seen_ == 64
    with (folder / 'train.csv').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='modificati'):
        run_training(config, 'ridge', tmp_path / 'invalid')


def test_cli_refit_is_loadable_in_another_process(tmp_path):
    import subprocess
    import sys
    source = tmp_path / 'source.csv'
    synthetic_dataset(source)
    folder = tmp_path / 'split'
    create_split({'source': str(source), 'output': str(folder), 'seed': 42,
                  'test_fraction': .2, 'cv_folds': 5})
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'split_dir': str(folder), 'features': ['mass_kg'],
                                  'importance_repeats': 2}), encoding='utf-8')
    output = tmp_path / 'cli_run'
    subprocess.run([sys.executable, '-m', 'src.train', '--config', str(config),
                    '--model', 'ridge', '--output', str(output), '--refit'],
                   check=True, capture_output=True, text=True, timeout=60)
    code = ('import joblib,sys; a=joblib.load(sys.argv[1]); '
            'x=a["preprocessor"].transform([{"gender":"F","mass_kg":None}]); '
            'assert len(a["model"].predict(x))==1')
    subprocess.run([sys.executable, '-c', code, str(output / 'model.joblib')],
                   check=True, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize('model', ['random_forest', 'xgboost'])
def test_forward_training_is_nested_and_never_scales_or_reads_holdout(tmp_path, monkeypatch, model):
    import src.train as module
    source = tmp_path / 'source.csv'
    synthetic_dataset(source)
    folder = tmp_path / 'split'
    create_split({'source': str(source), 'output': str(folder), 'seed': 42,
                  'test_fraction': .2, 'cv_folds': 5})
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.name in ('test.csv', 'split_manifest.json'):
            raise AssertionError('Accesso holdout')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    original_select = module.forward_select
    seen = []
    def tracked_select(rows, *args, **kwargs):
        seen.append({r['test_id'] for r in rows})
        return original_select(rows, *args, **kwargs)
    monkeypatch.setattr(module, 'forward_select', tracked_select)
    original_fit = FoldPreprocessor.fit
    def check_scaling(prep, rows):
        assert prep.scale is False
        assert len(rows) < 64
        return original_fit(prep, rows)
    monkeypatch.setattr(FoldPreprocessor, 'fit', check_scaling)
    config = {'split_dir': str(folder), 'features': ['mass_kg', 'gender'],
              'scaling': False, 'importance_repeats': 2,
              'feature_selection': {'method': 'forward', 'inner_folds': 3, 'min_improvement_W': .1},
              'models': {model: {'n_estimators': 5, 'max_depth': 2}}}
    output = tmp_path / 'forward'
    result = run_training(config, model, output)
    assert len(seen) == 5
    assert result['scaling'] is False
    assert result['refit'] is False
    with (output / 'predictions_oof.csv').open() as stream:
        oof = list(csv.DictReader(stream))
    for fold, train_ids in enumerate(seen):
        valid_ids = {r['test_id'] for r in oof if int(r['fold']) == fold}
        assert not train_ids & valid_ids
        assert len(train_ids | valid_ids) == 64
    audit = json.loads((output / 'feature_selection.json').read_text())
    assert len(audit) == 5
    assert all(set(r['selected_features']) <= set(config['features']) for r in audit)
    with (output / 'importance_folds.csv').open() as stream:
        importance = list(csv.DictReader(stream))
    with (output / 'importance.csv').open() as stream:
        aggregate = list(csv.DictReader(stream))
    for r in importance:
        expected = r['feature'] in audit[int(r['fold'])]['selected_features']
        assert (r['selected'] == 'True') == expected
        if not expected:
            assert float(r['mae_increase_W']) == 0
    for r in aggregate:
        values = [i for i in importance if i['feature'] == r['feature']]
        assert int(r['selected_folds']) == sum(i['selected'] == 'True' for i in values)
        assert float(r['mae_increase_W']) == pytest.approx(np.mean([float(i['mae_increase_W']) for i in values]))
    assert not (output / 'model.joblib').exists()


@pytest.mark.parametrize('model', ['random_forest', 'xgboost', 'bagging'])
@pytest.mark.parametrize('forward', [False, True])
def test_automatic_tuning_is_nested_and_preserves_holdout(tmp_path, monkeypatch, model, forward):
    import src.train as module
    source = tmp_path / 'source.csv'
    synthetic_dataset(source)
    folder = tmp_path / 'split'
    create_split({'source': str(source), 'output': str(folder), 'seed': 42,
                  'test_fraction': .2, 'cv_folds': 5})
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if path.name in ('test.csv', 'split_manifest.json'):
            raise AssertionError('Accesso holdout durante tuning')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    original_search = module.search_parameters
    searched_ids = []
    def tracked_search(rows, *args, **kwargs):
        searched_ids.append({r['test_id'] for r in rows})
        return original_search(rows, *args, **kwargs)
    monkeypatch.setattr(module, 'search_parameters', tracked_search)
    depth_key = 'estimator__max_depth' if model == 'bagging' else 'max_depth'
    config = {'split_dir': str(folder), 'features': ['mass_kg', 'gender'],
              'scaling': True, 'importance_repeats': 2,
              'models': {model: {'n_estimators': 3, depth_key: 2}},
              'tuning': {'method': 'random', 'n_iter': 2, 'inner_folds': 3,
                         'param_spaces': {model: {depth_key: [1, 2]}}}}
    if forward:
        config['feature_selection'] = {'method': 'forward', 'inner_folds': 3, 'min_improvement_W': .1}
    output = tmp_path / 'tuned'
    result = run_training(config, model, output)
    assert len(searched_ids) == 5
    assert result['refit'] is False and result['model_params'] is None
    assert not (output / 'model.joblib').exists()
    with (output / 'predictions_oof.csv').open() as stream:
        oof = list(csv.DictReader(stream))
    assert len(oof) == len({r['test_id'] for r in oof}) == 64
    audit = json.loads((output / 'tuning.json').read_text())
    for fold, record in enumerate(audit):
        valid_ids = {r['test_id'] for r in oof if int(r['fold']) == fold}
        assert not searched_ids[fold] & valid_ids
        assert len(searched_ids[fold] | valid_ids) == 64
        assert len(record['trials']) == 2
        assert record['inner_mae_W'] == min(t['inner_mae_W'] for t in record['trials'])
        assert record['best_params'] == result['selected_params_by_fold'][fold]
        assert result['model_params_by_fold'][fold][depth_key] == record['best_params'][depth_key]
        for trial in record['trials']:
            ids = {r['test_id'] for r in trial['selection']['inner_folds']}
            assert ids == searched_ids[fold]
            assert (output / f'tuning_fold_{fold}' / f'trial_{trial["trial"]:03d}.json').exists()
        assert record['trials'][0]['selection']['inner_folds'] == record['trials'][1]['selection']['inner_folds']


def test_tuning_refit_repeats_search_and_persists_chosen_parameters(tmp_path, monkeypatch):
    import joblib
    import src.train as module
    source = tmp_path / 'source.csv'
    synthetic_dataset(source)
    folder = tmp_path / 'split'
    create_split({'source': str(source), 'output': str(folder), 'seed': 42,
                  'test_fraction': .2, 'cv_folds': 5})
    original_search = module.search_parameters
    sizes = []
    def tracked_search(rows, *args, **kwargs):
        sizes.append(len(rows))
        return original_search(rows, *args, **kwargs)
    monkeypatch.setattr(module, 'search_parameters', tracked_search)
    config = {'split_dir': str(folder), 'features': ['mass_kg'], 'importance_repeats': 2,
              'tuning': {'method': 'random', 'n_iter': 2,
                         'param_spaces': {'ridge': {'alpha': [.01, 1.]}}}}
    output = tmp_path / 'refit_tuned'
    result = run_training(config, 'ridge', output, refit=True)
    assert len(sizes) == 6 and sizes[-1] == 64 and all(n < 64 for n in sizes[:-1])
    audit = json.loads((output / 'tuning_refit.json').read_text())
    artifact = joblib.load(output / 'model.joblib')
    assert artifact['model'].alpha == audit['best_params'].get('alpha', 1.)
    assert result['refit_params']['alpha'] == artifact['model'].alpha
    assert artifact['preprocessor'].scaler.n_samples_seen_ == 64


def test_mismatched_inner_folds_rejected_before_reading_data(tmp_path):
    with pytest.raises(ValueError, match='stessi inner_folds'):
        run_training({'features': ['mass_kg'],
                      'feature_selection': {'method': 'forward', 'inner_folds': 3},
                      'tuning': {'method': 'random', 'inner_folds': 4,
                                 'param_spaces': {'ridge': {'alpha': [1, 10]}}}},
                     'ridge', tmp_path / 'invalid')
