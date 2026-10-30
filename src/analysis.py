"""EDA descrittiva a livello di test, senza imputazione."""

import math

import numpy as np

from .dataset import FIELDS, SIGNALS, active_steps


def feature_metadata():
    fields = [{'key': k, 'label': label, 'unit': unit, 'group': group} for k, label, unit, group in FIELDS]
    fields += [
        {'key': 'age_years', 'label': 'Età al test', 'unit': 'anni', 'group': 'antropometria'},
        {'key': 'P_TLac_W_kg', 'label': 'T_Lac / massa', 'unit': 'W/kg', 'group': 'lattato'},
        {'key': 'active_steps_n', 'label': 'Step attivi', 'unit': 'n', 'group': 'segnali'},
        {'key': 'max_power_W', 'label': 'Potenza massima osservata', 'unit': 'W', 'group': 'segnali'},
    ]
    fields += [{'key': f'median_{k}', 'label': f'Mediana attiva {label}', 'unit': unit,
                'group': 'lattato' if k == 'bla_mmol_l' else 'segnali'} for k, label, unit in SIGNALS[1:]]
    return fields


def correlation_matrix(rows, keys, method):
    if method not in ('pearson', 'spearman'):
        raise ValueError(method)
    z, counts = [], []
    for x in keys:
        values, ns = [], []
        for y in keys:
            pairs = [(r.get(x), r.get(y)) for r in rows if r.get(x) is not None and r.get(y) is not None]
            ns.append(len(pairs))
            coefficient = None
            if len(pairs) >= 3:
                a, b = np.array(pairs, dtype=float).T
                if method == 'spearman':
                    a, b = average_ranks(a), average_ranks(b)
                if np.ptp(a) > 0 and np.ptp(b) > 0:
                    value = float(np.corrcoef(a, b)[0, 1])
                    coefficient = value if math.isfinite(value) else None
            values.append(coefficient)
        z.append(values)
        counts.append(ns)
    return {'z': z, 'n': counts}


def average_ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - (counts - 1) / 2
    return ranks[inverse]


def describe(values, total):
    clean = np.array([v for v in values if v is not None], dtype=float)
    return {'n': len(clean), 'missing': total - len(clean),
            **{label: float(fn()) if len(clean) else None for label, fn in [
                ('min', clean.min), ('q1', lambda: np.quantile(clean, .25)), ('median', lambda: np.median(clean)),
                ('q3', lambda: np.quantile(clean, .75)), ('max', clean.max), ('mean', clean.mean)]}}


def flag_extremes(records):
    """Segnalazione descrittiva 3×IQR, senza esclusione né giudizio fisiologico."""
    for field in feature_metadata():
        key = field['key']
        values = [r['features'].get(key) for r in records if r['features'].get(key) is not None]
        if len(values) < 4:
            continue
        q1, q3 = np.quantile(values, [.25, .75])
        spread = q3 - q1
        if spread == 0:
            continue
        lo, hi = q1 - 3 * spread, q3 + 3 * spread
        for record in records:
            value = record['features'].get(key)
            if value is not None and not lo <= value <= hi:
                record['issues'].append(f'Estremo statistico {key}={value:g}: fuori [Q1−3×IQR, Q3+3×IQR]; conservato')


def analyze(records):
    fields = feature_metadata()
    rows = [r['features'] for r in records]
    keys = [f['key'] for f in fields]
    statistics = {key: describe([r.get(key) for r in rows], len(rows)) for key in keys}
    active = [step for r in records for step in active_steps(r['steps'])]
    signal_stats = {k: describe([s[k] for s in active], len(active)) for k, _, _ in SIGNALS}
    return {
        'fields': fields, 'stats': statistics, 'signal_stats': signal_stats,
        'correlations': {m: correlation_matrix(rows, keys, m) for m in ('pearson', 'spearman')},
        'active_n': len(active), 'rest_n': sum(s['power_W'] == 0 for r in records for s in r['steps']),
    }
