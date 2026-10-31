"""Training dedicato: baseline proporzionale e regressione degli scostamenti."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error

from ..build_features import FEATURES
from ..first_model.train import MODELS, effective_params, load_training, make_model, write_json
from ..first_model.tuning import sample_candidates
from ..split_dataset import csv_bytes, make_folds
from .model import OffsetRegressor, estimate_k, maximum_power, validate_proportional
from .reporting import comparison_metrics, write_report


def validate_config(config, model_name):
    allowed = {'split_dir', 'seed', 'features', 'models', 'scaling', 'target_mode',
               'proportional', 'tuning', 'importance_repeats'}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError('Chiavi di configurazione non supportate nella fase residuals')
    features = config.get('features')
    if (not isinstance(features, list) or not features
            or any(not isinstance(f, str) for f in features)
            or len(set(features)) != len(features)
            or not set(features) <= ({'gender', *FEATURES} - {'max_power_W'})):
        raise ValueError('Lista feature non valida; esclusi target, lattato e max_power_W')
    if model_name not in MODELS:
        raise ValueError('Modello non supportato')
    if not isinstance(config.get('split_dir'), str):
        raise ValueError('split_dir richiesto')
    seed = config.get('seed', 42)
    if type(seed) is not int or seed < 0:
        raise ValueError('seed deve essere intero non negativo')
    scaling = config.get('scaling', model_name == 'ridge')
    if type(scaling) is not bool:
        raise ValueError('scaling deve essere booleano')
    mode = config.get('target_mode', 'residual')
    if mode not in ('residual', 'direct'):
        raise ValueError('target_mode deve essere residual o direct')
    proportional = config.get('proportional', {'mode': 'fit'})
    validate_proportional(proportional)
    repeats = config.get('importance_repeats', 5)
    if type(repeats) is not int or repeats < 2:
        raise ValueError('importance_repeats deve essere almeno 2')
    params = config.get('models', {}).get(model_name, {})
    make_model(model_name, params, seed)
    tuning = config.get('tuning')
    candidates = ([params] if tuning is None else sample_candidates(
        params, tuning, model_name, seed,
        canonicalize=lambda p: effective_params(model_name, p, seed)))
    if tuning is not None:
        for key, values in tuning['param_spaces'][model_name].items():
            for value in values:
                make_model(model_name, {**params, key: value}, seed)
    for candidate in candidates:
        make_model(model_name, candidate, seed)
    return {**config, 'seed': seed, 'scaling': scaling, 'target_mode': mode,
            'proportional': proportional, 'importance_repeats': repeats}, candidates


def make_predictor(config, model_name, params):
    return OffsetRegressor(config['features'], model_name, params,
                           scaling=config['scaling'], seed=config['seed'],
                           target_mode=config['target_mode'], proportional=config['proportional'])


def tune(rows, config, model_name, candidates, output, label):
    """Riceve solo il train esterno. Target originali: k si ristima in ogni fit."""
    if config.get('tuning') is None:
        return candidates[0], {'trials': [], 'partitions': [], 'best_params': candidates[0]}
    count = config['tuning']['inner_folds'] if 'inner_folds' in config['tuning'] else 3
    assignment = make_folds(rows, count, config['seed'])
    by_id = {r['test_id']: r['fold'] for r in assignment}
    partitions, audits = [], []
    for fold in range(count):
        train = [r for r in rows if by_id[r['test_id']] != fold]
        valid = [r for r in rows if by_id[r['test_id']] == fold]
        partitions.append((train, valid))
        audits.append({'fold': fold, 'train_ids': [r['test_id'] for r in train],
                       'validation_ids': [r['test_id'] for r in valid],
                       'k': estimate_k(train, config['proportional'])})
    trials = []
    trial_dir = output / f'tuning_{label}'
    trial_dir.mkdir()
    for index, params in enumerate(candidates):
        scores, base_scores = [], []
        for train, valid in partitions:
            predictor = make_predictor(config, model_name, params).fit(train)
            baseline, _, prediction = predictor.predict_parts(valid)
            y = np.array([r['P_TLac_W'] for r in valid])
            scores.append(float(mean_absolute_error(y, prediction)))
            base_scores.append(float(mean_absolute_error(y, baseline)))
        trial = {'trial': index, 'params': params, 'inner_mae_W': float(np.mean(scores)),
                 'fold_mae_W': scores, 'baseline_fold_mae_W': base_scores}
        trials.append(trial)
        write_json(trial_dir / f'trial_{index:03d}.json', trial)
        print(f'{label}: {index+1}/{len(candidates)}, MAE interno {trial["inner_mae_W"]:.3f} W', flush=True)
    best = min(trials, key=lambda t: t['inner_mae_W'])
    return best['params'], {'partitions': audits, 'trials': trials,
                            'best_trial': best['trial'], 'best_params': best['params']}


def run_training(config, model_name, output, refit=False):
    config, candidates = validate_config(config, model_name)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows, folds, manifest, columns = load_training(Path(config['split_dir']))
    if not set(config['features']) <= set(columns):
        raise ValueError('feature richieste mancanti dal train')
    maximum_power(rows)  # Fallire prima di creare artefatti; mai imputare P_max.
    if any(sum(folds[r['test_id']] == f for r in rows) < 2 for f in set(folds.values())):
        raise ValueError('Validation troppo piccola per R2')
    output.mkdir(parents=True)
    write_json(output / 'config.json', config)
    write_json(output / 'tuning_candidates.json', candidates)
    records, fold_audit, fold_metrics, importance = [], [], [], []
    for fold in sorted(set(folds.values())):
        train = [r for r in rows if folds[r['test_id']] != fold]
        valid = [r for r in rows if folds[r['test_id']] == fold]
        params, tuning = tune(train, config, model_name, candidates, output, f'fold_{fold}')
        predictor = make_predictor(config, model_name, params).fit(train)
        baseline, delta_hat, prediction = predictor.predict_parts(valid)
        y = np.array([r['P_TLac_W'] for r in valid])
        comparison = comparison_metrics(y, baseline, prediction)
        fold_metrics.append({'fold': fold, 'n_validation': len(valid), 'k': predictor.k_,
                             **comparison})
        fold_audit.append({'fold': fold, 'train_ids': [r['test_id'] for r in train],
                           'validation_ids': [r['test_id'] for r in valid],
                           'params': effective_params(model_name, params, config['seed']),
                           **predictor.metadata(), 'tuning': tuning})
        for i, row in enumerate(valid):
            records.append({'test_id': row['test_id'], 'fold': fold, 'k': predictor.k_,
                            'max_power_W': row['max_power_W'], 'observed_W': float(y[i]),
                            'baseline_W': float(baseline[i]), 'observed_delta_W': float(y[i]-baseline[i]),
                            'predicted_delta_W': float(delta_hat[i]), 'predicted_W': float(prediction[i]),
                            'error_predicted_minus_observed_W': float(prediction[i]-y[i])})
        # Il riferimento resta invariato: P_max non e una feature permutabile.
        rng = np.random.default_rng(config['seed'] + fold)
        base_mae = comparison['corrected']['mae_W']
        for feature in config['features']:
            changes = []
            for _ in range(config['importance_repeats']):
                order = rng.permutation(len(valid))
                shuffled = [{**row, feature: valid[order[i]][feature]} for i, row in enumerate(valid)]
                changes.append(mean_absolute_error(y, predictor.predict(shuffled)) - base_mae)
            importance.append({'fold': fold, 'feature': feature,
                               'mae_increase_W': float(np.mean(changes)),
                               'repeat_std_W': float(np.std(changes))})
        write_json(output / 'folds.json', fold_audit)
        write_json(output / 'metrics_folds.json', fold_metrics)
    order = {r['test_id']: i for i, r in enumerate(rows)}
    records.sort(key=lambda r: order[r['test_id']])
    assert len(records) == len({r['test_id'] for r in records}) == len(rows)
    observed = np.array([r['observed_W'] for r in records])
    baseline = np.array([r['baseline_W'] for r in records])
    prediction = np.array([r['predicted_W'] for r in records])
    result = {'model': model_name, 'target_mode': config['target_mode'],
              'proportional': config['proportional'], 'features': config['features'],
              'n_oof': len(rows), 'folds': manifest['cv_folds'], 'refit': refit,
              'scaling': config['scaling'], 'test_accessed': False,
              'oof': comparison_metrics(observed, baseline, prediction),
              'train_sha256': manifest['train_sha256'], 'folds_sha256': manifest['folds_sha256'],
              'versions': {p: version(p) for p in ['numpy', 'scikit-learn', 'xgboost', 'joblib']}}
    tables = {'predictions_oof.csv': records, 'importance_folds.csv': importance}
    tables['importance.csv'] = [{'feature': f,
        'mae_increase_W': float(np.mean([r['mae_increase_W'] for r in importance if r['feature'] == f])),
        'fold_std_W': float(np.std([r['mae_increase_W'] for r in importance if r['feature'] == f], ddof=1))}
        for f in config['features']]
    def flatten(comparison, scope):
        return [{'scope': scope, 'procedure': name, **comparison[name]}
                for name in ['proportional', 'corrected', 'delta_zero', 'delta_model']]
    tables['metrics.csv'] = flatten(result['oof'], 'oof')
    tables['metrics_folds.csv'] = [row for f in fold_metrics for row in flatten(f, f['fold'])]
    tables['gains_folds.csv'] = [{k: f[k] for k in ['fold', 'n_validation', 'k', 'q_delta',
                                'mae_gain_W', 'rmse_gain_W', 'delta_correlation']} for f in fold_metrics]
    for filename, table in tables.items():
        (output / filename).write_bytes(csv_bytes(table, list(table[0])))
    if refit:
        params, audit = tune(rows, config, model_name, candidates, output, 'refit')
        final = make_predictor(config, model_name, params).fit(rows)
        write_json(output / 'tuning_refit.json', audit)
        result['refit_params'] = effective_params(model_name, params, config['seed'])
        result['refit_preprocessing'] = final.metadata()
        joblib.dump({'predictor': final, 'features': config['features'],
                     'train_sha256': manifest['train_sha256'], 'target_mode': config['target_mode'],
                     'required_at_inference': ['max_power_W', 'gender', *[f for f in config['features'] if f != 'gender']]},
                    output / 'model.joblib')
    write_report(output, result, records)
    write_json(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--model', required=True, choices=sorted(MODELS))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--target-mode', choices=['residual', 'direct'], help='Override esplicito salvato nel config della run')
    parser.add_argument('--refit', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.target_mode:
        config['target_mode'] = args.target_mode
    result = run_training(config, args.model, args.output, args.refit)
    print(json.dumps({'model': result['model'], 'oof': result['oof'], 'refit': result['refit']}, indent=2))


if __name__ == '__main__':
    main()
