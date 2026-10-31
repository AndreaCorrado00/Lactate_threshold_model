"""Ricerca iperparametri esclusivamente sui train esterni e CV interna."""

import itertools
import json
import math

import numpy as np
from sklearn.metrics import mean_absolute_error

from .feature_selection import forward_select
from .preprocessing import FoldPreprocessor
from ..split_dataset import make_folds


# Per questi parametri 1 e 1.0 hanno la stessa semantica. Non normalizzare
# max_features/max_samples/min_samples_leaf: interi e frazioni possono differire.
CONTINUOUS_PARAMS = {
    'alpha', 'tol', 'learning_rate', 'reg_lambda', 'reg_alpha', 'subsample',
    'colsample_bytree', 'colsample_bylevel', 'colsample_bynode', 'min_child_weight',
    'gamma', 'min_impurity_decrease', 'ccp_alpha', 'min_weight_fraction_leaf',
}


def sample_candidates(base, config, model_name, seed, canonicalize=None):
    allowed = {'method', 'n_iter', 'inner_folds', 'param_spaces'}
    if not isinstance(config, dict) or set(config) - allowed or config.get('method') != 'random':
        raise ValueError('tuning richiede method=random e chiavi supportate')
    n_iter = config.get('n_iter', 20)
    inner_folds = config.get('inner_folds', 3)
    if type(n_iter) is not int or n_iter < 1:
        raise ValueError('n_iter deve essere un intero positivo')
    if type(inner_folds) is not int or inner_folds < 2:
        raise ValueError('inner_folds deve essere almeno 2')
    spaces = config.get('param_spaces')
    if not isinstance(spaces, dict) or model_name not in spaces:
        raise ValueError(f'param_spaces mancante per {model_name}')
    space = spaces[model_name]
    if not isinstance(space, dict) or not space:
        raise ValueError('Spazio di ricerca vuoto o non valido')
    for name, values in space.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f'Lista non vuota richiesta per {name}')
        if any(not isinstance(v, (str, int, float, bool, type(None)))
               or (isinstance(v, float) and not math.isfinite(v)) for v in values):
            raise ValueError(f'Valori non validi nello spazio: {name}')
    if math.prod(len(v) for v in space.values()) > 100_000:
        raise ValueError('Spazio troppo ampio: limitare a 100000 combinazioni finite')
    keys = sorted(space)
    def signature(params):
        normalized = canonicalize(params) if canonicalize is not None else params
        normalized = {k: float(v) if k in CONTINUOUS_PARAMS and type(v) in (int, float) else v
                      for k, v in normalized.items()}
        return json.dumps(normalized, sort_keys=True, allow_nan=False)
    seen = {signature(base)}
    pool = []
    for values in itertools.product(*(space[k] for k in keys)):
        params = {**base, **dict(zip(keys, values))}
        identity = signature(params)
        if identity not in seen:
            seen.add(identity)
            pool.append(params)
    order = np.random.default_rng(seed).permutation(len(pool))
    return [dict(base), *[pool[i] for i in order[:n_iter - 1]]]


def score_fixed_features(rows, features, factory, scaling, seed, n_folds, pca=None):
    assignments = make_folds(rows, n_folds, seed)
    fold_by_id = {r['test_id']: r['fold'] for r in assignments}
    scores = []
    for fold in range(n_folds):
        train = [r for r in rows if fold_by_id[r['test_id']] != fold]
        valid = [r for r in rows if fold_by_id[r['test_id']] == fold]
        prep = FoldPreprocessor(features, scale=scaling, pca=pca).fit(train)
        estimator = factory()
        estimator.fit(prep.transform(train), [r['P_TLac_W'] for r in train])
        prediction = estimator.predict(prep.transform(valid))
        scores.append(float(mean_absolute_error([r['P_TLac_W'] for r in valid], prediction)))
    return {'selected_features': features, 'inner_mae_W': float(np.mean(scores)),
            'fold_mae_W': scores, 'inner_folds': assignments}


def search_parameters(rows, features, candidates, model_factory, scaling, seed,
                      inner_folds, selection=None, on_trial=None, pca=None):
    """La validation esterna non viene passata a questa funzione.

    I punteggi interni sono usati per scegliere, non per riportare performance
    indipendenti. Le partizioni interne sono identiche per tutti i candidati.
    """
    if not candidates:
        raise ValueError('Nessuna configurazione candidata')
    if pca is not None and selection is not None:
        raise ValueError('PCA e forward selection non combinabili')
    trials = []
    for index, params in enumerate(candidates):
        factory = lambda: model_factory(params)
        try:
            if selection is not None:
                chosen, details = forward_select(
                    rows, features, factory, scaling, seed, inner_folds,
                    selection.get('min_improvement_W', .1))
            else:
                chosen = list(features)
                details = score_fixed_features(rows, chosen, factory, scaling, seed, inner_folds, pca=pca)
            if not np.isfinite(details['inner_mae_W']):
                raise ValueError('MAE interno non finito')
        except Exception as error:
            raise ValueError(f'Tuning fallito al tentativo {index}, parametri {params}: {error}') from error
        trial = {'trial': index, 'params': params, 'selected_features': chosen,
                 'inner_mae_W': details['inner_mae_W'], 'selection': details}
        trials.append(trial)
        if on_trial is not None:
            on_trial(trial)
    best = min(trials, key=lambda r: r['inner_mae_W'])
    return best['params'], best['selected_features'], {
        'best_trial': best['trial'], 'best_params': best['params'],
        'selected_features': best['selected_features'],
        'inner_mae_W': best['inner_mae_W'], 'trials': trials}
