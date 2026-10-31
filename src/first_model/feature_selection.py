"""Selezione sequenziale forward su CV interna, senza validation esterna."""

import numpy as np
from sklearn.metrics import mean_absolute_error

from .preprocessing import FoldPreprocessor
from ..split_dataset import make_folds


def forward_select(rows, features, model_factory, scaling, seed, inner_folds=3,
                   min_improvement_W=.1):
    assignments = make_folds(rows, inner_folds, seed)
    fold_by_id = {r['test_id']: r['fold'] for r in assignments}
    partitions = []
    for fold in range(inner_folds):
        train = [r for r in rows if fold_by_id[r['test_id']] != fold]
        valid = [r for r in rows if fold_by_id[r['test_id']] == fold]
        partitions.append((train, valid))
    selected, trace = [], []
    current_mae = None
    stop_reason = 'tutte le feature selezionate'
    while len(selected) < len(features):
        candidates = []
        for feature in features:
            if feature in selected:
                continue
            subset = [*selected, feature]
            scores = []
            for train, valid in partitions:
                prep = FoldPreprocessor(subset, scale=scaling).fit(train)
                estimator = model_factory()
                estimator.fit(prep.transform(train), [r['P_TLac_W'] for r in train])
                predicted = estimator.predict(prep.transform(valid))
                scores.append(float(mean_absolute_error([r['P_TLac_W'] for r in valid], predicted)))
            mean_mae = float(np.mean(scores))
            if not np.isfinite(mean_mae):
                raise ValueError('MAE interno non finito')
            candidate = {'step': len(selected) + 1, 'added_feature': feature,
                         'features': subset, 'inner_mae_W': mean_mae,
                         'fold_mae_W': scores, 'accepted': False}
            candidates.append(candidate)
        # min conserva l'ordine di configurazione nei pareggi esatti.
        best = min(candidates, key=lambda r: r['inner_mae_W'])
        improvement = None if current_mae is None else current_mae - best['inner_mae_W']
        accepted = improvement is None or (improvement > 0 and improvement >= min_improvement_W)
        best['accepted'] = accepted
        trace.extend(candidates)
        if not accepted:
            stop_reason = 'miglioramento insufficiente'
            break
        selected = best['features']
        current_mae = best['inner_mae_W']
    return selected, {'selected_features': selected, 'inner_mae_W': current_mae,
                      'stop_reason': stop_reason, 'candidates': trace,
                      'inner_folds': assignments}
