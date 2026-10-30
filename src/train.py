"""Cross-validation sul solo train; refit completo esclusivamente con --refit."""

import argparse
import csv
import io
import json
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from .build_features import FEATURES
from .preprocessing import FoldPreprocessor
from .feature_selection import forward_select
from .tuning import sample_candidates, search_parameters
from .split_dataset import csv_bytes, digest, parse_rows, stratum


MODELS = {'ridge': Ridge, 'random_forest': RandomForestRegressor,
          'xgboost': XGBRegressor, 'bagging': BaggingRegressor}


def make_model(name, params, seed):
    if name not in MODELS:
        raise ValueError(f'Modello non supportato: {name}')
    forbidden = {'random_state', 'seed', 'n_jobs', 'nthread', 'early_stopping_rounds', 'callbacks'}
    if forbidden & set(params):
        raise ValueError('Seed/thread gestiti dalla pipeline; early stopping non previsto')
    if name == 'bagging':
        if 'estimator' in params or any(k.split('__')[-1] in forbidden for k in params):
            raise ValueError('Bagging: albero base e seed gestiti dalla pipeline')
        estimator = BaggingRegressor(estimator=DecisionTreeRegressor(random_state=seed),
                                     random_state=seed, n_jobs=1)
        if set(params) - set(estimator.get_params(deep=True)):
            raise ValueError('Iperparametri Bagging sconosciuti; usare estimator__ per l’albero')
        return estimator.set_params(**params)
    valid = set(MODELS[name]().get_params())
    if set(params) - valid:
        raise ValueError(f'Iperparametri sconosciuti: {sorted(set(params) - valid)}')
    fixed = {'random_state': seed}
    if name != 'ridge':
        fixed['n_jobs'] = 1
    if name == 'xgboost':
        fixed.update(objective='reg:squarederror', tree_method='hist', device='cpu')
        if {'objective', 'tree_method', 'device'} & set(params):
            raise ValueError('Obiettivo regressione e backend CPU fissati dalla pipeline')
    return MODELS[name](**params, **fixed)


def metrics(y, prediction):
    return {'mae_W': float(mean_absolute_error(y, prediction)),
            'rmse_W': float(np.sqrt(mean_squared_error(y, prediction))),
            'r2': float(r2_score(y, prediction))}


def load_training(split_dir):
    """Non legge né stat né contenuto di test.csv o split_manifest.json."""
    manifest = json.loads((split_dir / 'train_manifest.json').read_text(encoding='utf-8'))
    content = (split_dir / 'train.csv').read_bytes()
    fold_content = (split_dir / 'cv_folds.csv').read_bytes()
    if digest(content) != manifest['train_sha256'] or digest(fold_content) != manifest['folds_sha256']:
        raise ValueError('Train o assegnazioni fold modificati dopo lo split')
    rows, columns = parse_rows(content)
    assignments = list(csv.DictReader(io.StringIO(fold_content.decode('utf-8'))))
    if len(rows) != manifest['n_train'] or len(assignments) != len(rows):
        raise ValueError('Numerosità train/fold incoerente')
    folds = {r['test_id']: int(r['fold']) for r in assignments}
    if len(folds) != len(rows) or set(folds) != {r['test_id'] for r in rows}:
        raise ValueError('ID dei fold non corrispondenti al train')
    if set(folds.values()) != set(range(manifest['cv_folds'])):
        raise ValueError('Fold mancanti o non validi')
    labels = {r['test_id']: r['stratum'] for r in assignments}
    if any(labels[r['test_id']] != stratum(r) for r in rows):
        raise ValueError('Strati dei fold incoerenti')
    return rows, folds, manifest, columns


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def effective_params(model_name, params, seed):
    def serial(value):
        if isinstance(value, BaseEstimator):
            return {'class': type(value).__name__, 'params': serial(value.get_params(deep=False))}
        if isinstance(value, dict):
            return {k: serial(v) for k, v in value.items()}
        if isinstance(value, float) and np.isnan(value):
            return 'NaN'
        return value
    return serial(make_model(model_name, params, seed).get_params())


def tune_on_training(rows, features, candidates, model_name, scaling, seed,
                     n_folds, selection, output, label, pca=None):
    trial_dir = output / f'tuning_{label}'
    trial_dir.mkdir()
    def save_trial(trial):
        write_json(trial_dir / f'trial_{trial["trial"]:03d}.json', trial)
        print(f'  {label}: tentativo {trial["trial"] + 1}/{len(candidates)}, '
              f'MAE interno {trial["inner_mae_W"]:.3f} W, '
              f'{len(trial["selected_features"])} feature', flush=True)
    return search_parameters(rows, features, candidates,
                             lambda p: make_model(model_name, p, seed),
                             scaling, seed, n_folds, selection, save_trial, pca=pca)


def run_training(config, model_name, output, refit=False):
    features = config.get('features', [])
    allowed = {'gender', *FEATURES}
    if not isinstance(features, list) or not features or any(not isinstance(k, str) for k in features):
        raise ValueError('Lista feature richiesta')
    if len(features) != len(set(features)) or not set(features) <= allowed:
        raise ValueError('feature duplicate, sconosciute o non consentite')
    if set(config) - {'features', 'split_dir', 'seed', 'importance_repeats', 'models', 'scaling', 'feature_selection', 'tuning', 'pca'}:
        raise ValueError('Chiavi configurazione training sconosciute')
    if model_name not in MODELS:
        raise ValueError('Modello non supportato')
    seed = config.get('seed', 42)
    scaling = config.get('scaling', model_name == 'ridge')
    if not isinstance(scaling, bool):
        raise ValueError('scaling deve essere booleano')
    selection = config.get('feature_selection')
    pca = config.get('pca')
    FoldPreprocessor(features, scale=scaling, pca=pca)  # Validazione senza accesso ai dati.
    if pca is not None and selection is not None:
        raise ValueError('PCA richiede feature_selection=null')
    if selection is not None:
        if not isinstance(selection, dict) or set(selection) - {'method', 'inner_folds', 'min_improvement_W'}:
            raise ValueError('Configurazione feature_selection non valida')
        if selection.get('method') != 'forward':
            raise ValueError('Metodo di feature selection non supportato')
        inner_folds = selection.get('inner_folds', 3)
        improvement = selection.get('min_improvement_W', .1)
        if not isinstance(inner_folds, int) or inner_folds < 2:
            raise ValueError('inner_folds deve essere almeno 2')
        if not isinstance(improvement, (int, float)) or not np.isfinite(improvement) or improvement < 0:
            raise ValueError('min_improvement_W deve essere finito e non negativo')
    repeats = config.get('importance_repeats', 10)
    if not isinstance(repeats, int) or repeats < 2:
        raise ValueError('importance_repeats deve essere almeno 2')
    params = config.get('models', {}).get(model_name, {})
    make_model(model_name, params, seed)  # Validazione prima di creare output.
    tuning = config.get('tuning')
    candidates = None
    if tuning is not None:
        candidates = sample_candidates(params, tuning, model_name, seed,
                                       canonicalize=lambda p: effective_params(model_name, p, seed))
        tuning_folds = tuning.get('inner_folds', 3)
        if selection is not None and inner_folds != tuning_folds:
            raise ValueError('Tuning e forward selection devono usare gli stessi inner_folds')
        # Controlla anche i nomi di parametri eventualmente non campionati.
        for key, values in tuning['param_spaces'][model_name].items():
            make_model(model_name, {**params, key: values[0]}, seed)
        for candidate in candidates:
            make_model(model_name, candidate, seed)
    rows, folds, manifest, columns = load_training(Path(config['split_dir']))
    if not set(features) <= set(columns):
        raise ValueError('feature richieste assenti dal train')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Output già presente: {output}')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    if tuning is not None:
        write_json(output / 'tuning_candidates.json', candidates)
    y = np.array([r['P_TLac_W'] for r in rows])
    prediction = np.full(len(rows), np.nan)
    fold_metrics, importance_rows, preprocessing, selection_audit = [], [], [], []
    tuning_audit, params_by_fold = [], []
    for fold in range(manifest['cv_folds']):
        valid_idx = np.array([i for i, r in enumerate(rows) if folds[r['test_id']] == fold])
        train_idx = np.array([i for i, r in enumerate(rows) if folds[r['test_id']] != fold])
        if len(valid_idx) < 2:
            raise ValueError('Validation fold troppo piccolo per R²')
        train = [rows[i] for i in train_idx]
        valid = [rows[i] for i in valid_idx]
        selected_features = features
        selected_params = params
        if tuning is not None:
            print(f'{model_name}: fold esterno {fold + 1}/{manifest["cv_folds"]}, '
                  f'tuning di {len(candidates)} configurazioni...', flush=True)
            selected_params, selected_features, audit = tune_on_training(
                train, features, candidates, model_name, scaling, seed,
                tuning_folds, selection, output, f'fold_{fold}', pca=pca)
            tuning_audit.append({'fold': fold, **audit})
            write_json(output / 'tuning.json', tuning_audit)
            if selection is not None:
                selection_audit.append({'fold': fold, **audit['trials'][audit['best_trial']]['selection']})
                write_json(output / 'feature_selection.json', selection_audit)
        elif selection is not None:
            print(f'{model_name}: fold esterno {fold + 1}/{manifest["cv_folds"]}, forward selection...', flush=True)
            selected_features, audit = forward_select(
                train, features, lambda: make_model(model_name, params, seed),
                scaling, seed, inner_folds, improvement)
            selection_audit.append({'fold': fold, **audit})
            (output / 'feature_selection.json').write_text(
                json.dumps(selection_audit, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            print(f'  Selezionate {len(selected_features)} feature: {", ".join(selected_features)}', flush=True)
        prep = FoldPreprocessor(selected_features, scale=scaling, pca=pca).fit(train)
        x_train, x_valid = prep.transform(train), prep.transform(valid)
        params_by_fold.append(selected_params)
        estimator = make_model(model_name, selected_params, seed)
        estimator.fit(x_train, y[train_idx])
        predicted = estimator.predict(x_valid)
        prediction[valid_idx] = predicted
        fold_metrics.append({'fold': fold, 'n_train': len(train_idx), 'n_validation': len(valid_idx),
                             'n_features': len(selected_features),
                             **metrics(y[valid_idx], predicted)})
        if pca is None:
            importance = permutation_importance(estimator, x_valid, y[valid_idx],
                                                scoring='neg_mean_absolute_error', n_repeats=repeats,
                                                random_state=seed + fold, n_jobs=1)
        else:
            # Permutazione delle covariate originali, prima di imputazione/scaling/PCA.
            rng = np.random.default_rng(seed + fold)
            base_mae = mean_absolute_error(y[valid_idx], predicted)
            increases = []
            for key in selected_features:
                scores = []
                for _ in range(repeats):
                    order = rng.permutation(len(valid))
                    shuffled = [{**row, key: valid[order[i]][key]} for i, row in enumerate(valid)]
                    scores.append(mean_absolute_error(y[valid_idx], estimator.predict(prep.transform(shuffled))) - base_mae)
                increases.append(scores)
            importance = SimpleNamespace(importances_mean=np.mean(increases, axis=1),
                                         importances_std=np.std(increases, axis=1))
        for key in features:
            chosen = key in selected_features
            j = selected_features.index(key) if chosen else None
            importance_rows.append({'fold': fold, 'feature': key, 'selected': chosen,
                                    'mae_increase_W': float(importance.importances_mean[j]) if chosen else 0.,
                                    'repeat_std_W': float(importance.importances_std[j]) if chosen else 0.})
        preprocessing.append({'fold': fold, 'medians': prep.imputer.medians,
                              'features': selected_features, 'pca': prep.pca_metadata(),
                              'scaler_mean': prep.scaler.mean_.tolist() if prep.scaler else None,
                              'scaler_scale': prep.scaler.scale_.tolist() if prep.scaler else None})
    if not np.isfinite(prediction).all():
        raise ValueError('Predizioni OOF mancanti o non finite')
    aggregate = {key: {'mean': float(np.mean([r[key] for r in fold_metrics])),
                       'std': float(np.std([r[key] for r in fold_metrics], ddof=1))}
                 for key in ('mae_W', 'rmse_W', 'r2')}
    importance_summary = []
    for key in features:
        values = [r['mae_increase_W'] for r in importance_rows if r['feature'] == key]
        importance_summary.append({'feature': key, 'mae_increase_W': float(np.mean(values)),
                                   'selected_folds': sum(r['selected'] for r in importance_rows if r['feature'] == key),
                                   'fold_std_W': float(np.std(values, ddof=1))})
    importance_summary.sort(key=lambda r: r['mae_increase_W'], reverse=True)
    result = {'model': model_name, 'primary_metric': 'mae_W', 'n_oof': len(rows),
              'folds': manifest['cv_folds'], 'fold_aggregate': aggregate, 'oof': metrics(y, prediction),
              'refit': refit, 'test_accessed': False, 'features': features,
              'scaling': scaling, 'feature_selection': selection, 'pca': pca,
              'tuning': tuning, 'selected_params_by_fold': params_by_fold,
              'model_params_by_fold': [effective_params(model_name, p, seed) for p in params_by_fold],
              'selected_features_by_fold': [r['features'] for r in preprocessing],
              'train_sha256': manifest['train_sha256'], 'folds_sha256': manifest['folds_sha256'],
              'small_strata': manifest['small_strata'],
              'model_params': effective_params(model_name, params, seed) if tuning is None else None,
              'base_model_params': effective_params(model_name, params, seed),
              'versions': {package: version(package) for package in ['numpy', 'scikit-learn', 'xgboost', 'joblib']}}
    oof_rows = [{'test_id': r['test_id'], 'fold': folds[r['test_id']], 'observed_W': y[i],
                 'predicted_W': prediction[i], 'residual_W': y[i] - prediction[i]} for i, r in enumerate(rows)]
    tables = {'metrics_folds.csv': fold_metrics, 'predictions_oof.csv': oof_rows,
              'importance_folds.csv': importance_rows, 'importance.csv': importance_summary,
              'metrics.csv': [{'metric': key, 'fold_mean': aggregate[key]['mean'],
                               'fold_std': aggregate[key]['std'], 'oof': result['oof'][key]} for key in aggregate]}
    for filename, table in tables.items():
        (output / filename).write_bytes(csv_bytes(table, list(table[0])))
    for filename, obj in [('config.json', config), ('preprocessing_folds.json', preprocessing)]:
        (output / filename).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Validation OOF: osservato/predetto', 'Residui OOF'])
    fig.add_trace(go.Scatter(x=y, y=prediction, mode='markers', name='Test ciclistici'), row=1, col=1)
    bounds = [float(min(y.min(), prediction.min())), float(max(y.max(), prediction.max()))]
    fig.add_trace(go.Scatter(x=bounds, y=bounds, mode='lines', name='Identità'), row=1, col=1)
    fig.add_trace(go.Scatter(x=prediction, y=y - prediction, mode='markers', name='Residui'), row=1, col=2)
    fig.update_xaxes(title_text='Osservato (W)', row=1, col=1)
    fig.update_yaxes(title_text='Predetto (W)', row=1, col=1)
    fig.update_xaxes(title_text='Predetto (W)', row=1, col=2)
    fig.update_yaxes(title_text='Osservato − predetto (W)', row=1, col=2)
    fig.write_html(output / 'validation.html', include_plotlyjs=True)
    importance_fig = go.Figure(go.Bar(x=[r['mae_increase_W'] for r in importance_summary],
                                      y=[r['feature'] for r in importance_summary], orientation='h',
                                      error_x={'type': 'data', 'array': [r['fold_std_W'] for r in importance_summary]}))
    importance_fig.update_layout(title='Permutation importance su validation: media ± SD tra fold',
                                 xaxis_title='Aumento MAE (W)', yaxis={'autorange': 'reversed'})
    importance_fig.write_html(output / 'importance.html', include_plotlyjs=True)
    if refit:
        final_features = features
        final_params = params
        if tuning is not None:
            final_params, final_features, audit = tune_on_training(
                rows, features, candidates, model_name, scaling, seed,
                tuning_folds, selection, output, 'refit', pca=pca)
            write_json(output / 'tuning_refit.json', audit)
            if selection is not None:
                write_json(output / 'feature_selection_refit.json', audit['trials'][audit['best_trial']]['selection'])
        elif selection is not None:
            final_features, audit = forward_select(
                rows, features, lambda: make_model(model_name, params, seed),
                scaling, seed, inner_folds, improvement)
            (output / 'feature_selection_refit.json').write_text(
                json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        prep = FoldPreprocessor(final_features, scale=scaling, pca=pca).fit(rows)
        result['refit_pca'] = prep.pca_metadata()
        estimator = make_model(model_name, final_params, seed).fit(prep.transform(rows), y)
        result['refit_params'] = effective_params(model_name, final_params, seed)
        result['refit_features'] = final_features
        joblib.dump({'preprocessor': prep, 'model': estimator, 'features': final_features,
                     'params': final_params,
                     'train_sha256': manifest['train_sha256']}, output / 'model.joblib')
    (output / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', choices=sorted(MODELS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--refit', action='store_true', help='Dopo CV, fit sul train completo; nessun accesso al test')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    result = run_training(config, args.model, args.output, refit=args.refit)
    print(json.dumps({'model': result['model'], 'validation': result['fold_aggregate'],
                      'oof': result['oof'], 'refit': result['refit']}, indent=2))


if __name__ == '__main__':
    main()
