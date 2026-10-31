"""Valuta un refit congelato; accesso al test solo con flag esplicito."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..split_dataset import csv_bytes, digest, parse_rows
from .train import metrics, write_json


def evaluate(run, holdout_dir, output, allow_test_access=False):
    if not allow_test_access:
        raise ValueError('Accesso vietato: richiede --allow-test-access e ordine esplicito dell utente')
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    if not summary.get('refit'):
        raise ValueError('La run non contiene un refit completo')
    # Caricare soltanto artefatti joblib locali e fidati prodotti dalla pipeline.
    model_path = run / 'model.joblib'
    bundle = joblib.load(model_path)
    train_dir = Path(config['split_dir'])
    train_content = (train_dir / 'train.csv').read_bytes()
    train_manifest = json.loads((train_dir / 'train_manifest.json').read_text(encoding='utf-8'))
    train_hash = digest(train_content)
    if any(value != train_hash for value in [summary['train_sha256'],
            bundle['train_sha256'], train_manifest['train_sha256']]):
        raise ValueError('Hash del train non coerente con il modello')
    train, _ = parse_rows(train_content)
    original_manifest = json.loads((holdout_dir / 'split_manifest.json').read_text(encoding='utf-8'))
    if original_manifest['train_sha256'] != train_manifest.get('parent_train_sha256', train_hash):
        raise ValueError('Holdout di provenienza diversa dal train')
    # Controllo anche contro tutti gli ID originali, inclusi quelli esclusi dal train QC.
    original_train_content = (holdout_dir / 'train.csv').read_bytes()
    if digest(original_train_content) != original_manifest['train_sha256']:
        raise ValueError('Hash del train originale non valido')
    original_train, _ = parse_rows(original_train_content)
    test_content = (holdout_dir / 'test.csv').read_bytes()
    if digest(test_content) != original_manifest['test_sha256']:
        raise ValueError('Hash del test non valido')
    test, columns = parse_rows(test_content)
    if len(test) != original_manifest['n_test'] or len(test) < 2:
        raise ValueError('Numerosita test non valida')
    train_ids = {r['test_id'] for r in train + original_train}
    if train_ids & {r['test_id'] for r in test}:
        raise ValueError('Sovrapposizione train/test')
    if not set(bundle['features']) <= set(columns):
        raise ValueError('Feature richieste mancanti nel test')
    if bundle['preprocessor'].features != bundle['features']:
        raise ValueError('Feature e preprocessing incoerenti')
    y = np.array([r['P_TLac_W'] for r in test])
    prediction = np.asarray(bundle['model'].predict(bundle['preprocessor'].transform(test)))
    if prediction.shape != y.shape or not np.all(np.isfinite(prediction)):
        raise ValueError('Predizioni non valide')
    scores = metrics(y, prediction)
    # Bootstrap descrittivo condizionato al modello congelato, per riga/test.
    rng = np.random.default_rng(42)
    samples = {key: [] for key in scores}
    for _ in range(2000):
        indices = rng.integers(0, len(y), len(y))
        values = metrics(y[indices], prediction[indices])
        for key, value in values.items():
            if key != 'r2' or np.ptp(y[indices]) > 0:
                samples[key].append(value)
    intervals = {key: np.quantile(values, [.025, .975]).tolist() if values else None
                 for key, values in samples.items()}
    result = {'n_test': len(test), 'metrics': scores, 'bootstrap_95_percent': intervals,
              'bootstrap_repeats': 2000, 'seed': 42, 'train_sha256': train_hash,
              'test_sha256': digest(test_content), 'model_sha256': digest(model_path.read_bytes()),
              'run': str(run), 'features': bundle['features'], 'test_accessed': True,
              'evaluation_only': True, 'bias_predicted_minus_observed_W': float(np.mean(prediction-y)),
              'bootstrap_limitation': 'Per test, modello fisso; non include incertezza del training o dipendenza tra test dello stesso atleta'}
    output.mkdir(parents=True, exist_ok=False)
    records = [{'test_id': r['test_id'], 'observed_W': float(y[i]),
                'predicted_W': float(prediction[i]), 'residual_W': float(y[i]-prediction[i])}
               for i, r in enumerate(test)]
    (output / 'predictions.csv').write_bytes(csv_bytes(records, list(records[0])))
    (output / 'metrics.csv').write_bytes(csv_bytes([scores], list(scores)))
    write_json(output / 'summary.json', result)
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Osservato e predetto', 'Residui'])
    fig.add_trace(go.Scatter(x=y, y=prediction, mode='markers', text=[r['test_id'] for r in test], name='Test'), row=1, col=1)
    limits = [float(min(y.min(), prediction.min())), float(max(y.max(), prediction.max()))]
    fig.add_trace(go.Scatter(x=limits, y=limits, mode='lines', name='Identita'), row=1, col=1)
    fig.add_trace(go.Scatter(x=prediction, y=y-prediction, mode='markers', text=[r['test_id'] for r in test], showlegend=False), row=1, col=2)
    fig.add_hline(y=0, row=1, col=2)
    fig.update_xaxes(type='linear', title_text='Osservato (W)', row=1, col=1)
    fig.update_yaxes(type='linear', title_text='Predetto (W)', row=1, col=1)
    fig.update_xaxes(type='linear', title_text='Predetto (W)', row=1, col=2)
    fig.update_yaxes(type='linear', title_text='Osservato - predetto (W)', row=1, col=2)
    fig.update_layout(title='Valutazione holdout: modello congelato')
    fig.write_html(output / 'validation.html', include_plotlyjs=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--holdout-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-test-access', action='store_true')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run, args.holdout_dir, args.output, args.allow_test_access), indent=2))


if __name__ == '__main__':
    main()
