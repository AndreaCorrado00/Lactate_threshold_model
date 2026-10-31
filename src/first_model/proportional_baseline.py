"""Confronto esplorativo OOF con P_TLac = k * P_max, sul solo train."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..split_dataset import csv_bytes
from .train import load_training, metrics, write_json


def proportional_oof(maximum, target, folds):
    maximum, target = np.asarray(maximum, dtype=float), np.asarray(target, dtype=float)
    folds = np.asarray(folds)
    if (maximum.ndim != 1 or maximum.shape != target.shape or folds.shape != target.shape
            or len(np.unique(folds)) < 2 or not np.all(np.isfinite(maximum))
            or not np.all(np.isfinite(target)) or np.any(maximum <= 0)):
        raise ValueError('Dati o fold non validi per il riferimento proporzionale')
    predictions = np.empty_like(target)
    coefficients = {}
    for fold in np.unique(folds):
        valid = folds == fold
        x, y = maximum[~valid], target[~valid]
        coefficient = float(np.dot(x, y) / np.dot(x, x))
        coefficients[int(fold)] = coefficient
        predictions[valid] = coefficient * maximum[valid]
    return predictions, coefficients


def run_comparison(split_dir, runs, output):
    rows, assignments, manifest, _ = load_training(split_dir)
    ids = [r['test_id'] for r in rows]
    y = np.array([r['P_TLac_W'] for r in rows])
    maximum = np.array([r['max_power_W'] for r in rows], dtype=float)
    folds = np.array([assignments[i] for i in ids])
    baseline, coefficients = proportional_oof(maximum, y, folds)
    predictions = {'proportional': baseline}
    for run in runs:
        summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
        if any(summary[key] != manifest[key] for key in ('train_sha256', 'folds_sha256')):
            raise ValueError(f'Partizioni diverse: {run}')
        with (run / 'predictions_oof.csv').open(encoding='utf-8-sig') as stream:
            records = list(csv.DictReader(stream))
        mapping = {r['test_id']: r for r in records}
        if len(records) != len(mapping) or set(mapping) != set(ids):
            raise ValueError(f'ID OOF non corrispondenti: {run}')
        if any(int(mapping[i]['fold']) != assignments[i] for i in ids):
            raise ValueError(f'Fold OOF non corrispondenti: {run}')
        if not np.allclose([float(mapping[i]['observed_W']) for i in ids], y):
            raise ValueError(f'Target OOF non corrispondenti: {run}')
        prediction = np.array([float(mapping[i]['predicted_W']) for i in ids])
        if not np.all(np.isfinite(prediction)) or run.name in predictions:
            raise ValueError(f'Predizioni non valide o run duplicata: {run}')
        predictions[run.name] = prediction
    aggregate, per_fold = [], []
    baseline_mae = metrics(y, baseline)['mae_W']
    for name, prediction in predictions.items():
        scores = metrics(y, prediction)
        aggregate.append({'model': name, **scores,
                          'delta_mae_vs_proportional_W': scores['mae_W'] - baseline_mae})
        for fold in sorted(coefficients):
            valid = folds == fold
            per_fold.append({'model': name, 'fold': fold, 'n_validation': int(valid.sum()),
                             **metrics(y[valid], prediction[valid])})
    # Solo nuovi artefatti; mai sovrascrivere una run precedente.
    output.mkdir(parents=True, exist_ok=False)
    for filename, records in [('metrics.csv', aggregate), ('metrics_folds.csv', per_fold)]:
        (output / filename).write_bytes(csv_bytes(records, list(records[0])))
    records = [{'test_id': i, 'fold': int(folds[j]), 'observed_W': float(y[j]),
                'max_power_W': float(maximum[j]),
                **{name: float(pred[j]) for name, pred in predictions.items()}}
               for j, i in enumerate(ids)]
    (output / 'predictions_oof.csv').write_bytes(csv_bytes(records, list(records[0])))
    write_json(output / 'summary.json', {
        'n_oof': len(rows), 'coefficients_by_fold': coefficients,
        'formula': 'k = sum(P_max * P_TLac) / sum(P_max ** 2), sui soli fold di training',
        'train_sha256': manifest['train_sha256'], 'folds_sha256': manifest['folds_sha256'],
        'runs': [str(p) for p in runs], 'test_accessed': False, 'refit': False})
    lines = ['# Confronto con il riferimento proporzionale', '',
             f'{len(rows)} test del train; stessi fold delle run confrontate. Nessun accesso al test set.', '',
             'Regressione senza intercetta, coefficiente stimato ai minimi quadrati solo sul training di ciascun fold.', '',
             '| Modello | MAE (W) | RMSE (W) | R2 | Delta MAE (W) |',
             '|---|---:|---:|---:|---:|']
    for row in aggregate:
        lines.append(f"| {row['model']} | {row['mae_W']:.3f} | {row['rmse_W']:.3f} | {row['r2']:.4f} | {row['delta_mae_vs_proportional_W']:+.3f} |")
    lines += ['', 'Delta MAE negativo: miglioramento rispetto al riferimento.', '',
              'Metriche aggregate sulle predizioni OOF; metriche per fold nel CSV dedicato.',
              'Confronto descrittivo esplorativo formulato dopo aver osservato i risultati: non costituisce un test confermativo di significativita.',
              'Gli ID identificano test, non necessariamente atleti indipendenti.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir', type=Path, default=Path('data/splits/v1'))
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_comparison(args.split_dir, args.runs, args.output), indent=2))


if __name__ == '__main__':
    main()
