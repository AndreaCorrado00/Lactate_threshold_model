"""PCA esplorativa sul solo train; nessuna valutazione predittiva."""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .build_features import FEATURES
from .preprocessing import FoldPreprocessor
from .split_dataset import csv_bytes
from .train import load_training, write_json


def build_pca_report(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    features = config['features']
    if not features or len(set(features)) != len(features) or not set(features) <= {'gender', *FEATURES}:
        raise ValueError('Feature non valide per PCA esplorativa')
    rows, _, manifest, columns = load_training(Path(config['split_dir']))
    if not set(features) <= set(columns):
        raise ValueError('Feature assenti dal train')
    numeric = [f for f in features if f != 'gender']
    n = min(len(rows), len(numeric))
    prep = FoldPreprocessor(features, scale=True, pca={'n_components': n}).fit(rows)
    meta = prep.pca_metadata()
    cumulative = np.cumsum(prep.pca_.explained_variance_ratio_)
    scores = prep.transform(rows)[:, :n]
    names = [f'PC{i+1}' for i in range(n)]
    variance = [{'component': name, 'explained_variance_ratio': float(prep.pca_.explained_variance_ratio_[i]),
                 'cumulative_variance_ratio': float(cumulative[i])} for i, name in enumerate(names)]
    weights = [{'feature': f, **{name: float(prep.pca_.components_[i, j]) for i, name in enumerate(names)}}
               for j, f in enumerate(numeric)]
    score_rows = [{'test_id': r['test_id'], 'gender': r['gender'],
                   **{name: float(scores[j, i]) for i, name in enumerate(names)}} for j, r in enumerate(rows)]
    output.mkdir(parents=True)
    for filename, table in [('variance.csv', variance), ('component_weights.csv', weights), ('scores.csv', score_rows)]:
        (output / filename).write_bytes(csv_bytes(table, list(table[0])))
    summary = {**meta, 'n_train': len(rows), 'test_accessed': False,
               'train_sha256': manifest['train_sha256'], 'features': features,
               'medians': prep.imputer.medians, 'scaler_mean': prep.scaler.mean_.tolist(),
               'scaler_scale': prep.scaler.scale_.tolist(),
               'components_for_variance': {str(v): int(min(n, np.searchsorted(cumulative, v)+1)) for v in [.8, .9, .95]},
               'scope': 'Esplorazione sul train completo; non usare questo fit nella CV'}
    write_json(output / 'summary.json', summary)
    write_json(output / 'config.json', config)
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Varianza spiegata', 'Pesi delle componenti (assi ortonormali)'])
    fig.add_trace(go.Bar(x=names, y=prep.pca_.explained_variance_ratio_, name='Quota'), row=1, col=1)
    fig.add_trace(go.Scatter(x=names, y=cumulative, name='Cumulata'), row=1, col=1)
    fig.add_trace(go.Heatmap(x=names, y=numeric, z=prep.pca_.components_.T,
                            colorscale='RdBu', zmid=0, showscale=False), row=1, col=2)
    fig.update_xaxes(type='category')
    fig.update_yaxes(type='linear', title_text='Quota di varianza', row=1, col=1)
    fig.update_yaxes(type='category', row=1, col=2)
    fig.update_layout(title='PCA esplorativa sul solo train', height=750)
    fig.write_html(output / 'pca.html', include_plotlyjs=True)
    (output / 'report.md').write_text(
        '# PCA esplorativa sul train\n\n'
        f'{len(rows)} test, {len(numeric)} covariate numeriche. Sesso escluso dalle componenti.\n\n'
        'Imputazione per sesso e standardizzazione stimate sul train completo, solo per esplorazione. '
        'Le configurazioni di training ristimano ogni trasformazione nei fold; non caricano questi artefatti.\n\n'
        f'Componenti per quota di varianza: {summary["components_for_variance"]}.\n\n'
        'I pesi sono gli autovettori sulle feature standardizzate, non correlazioni feature-componente '
        'e non importanze predittive. Il segno degli assi è arbitrario. '
        'Le componenti massimizzano la varianza delle covariate, non il legame con il target. '
        'Outlier e covariate ridondanti possono influenzarle.\n\n'
        '[Grafico](pca.html) · [Varianza](variance.csv) · [Pesi](component_weights.csv) · [Punteggi](scores.csv)\n',
        encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    build_pca_report(json.loads(args.config.read_text(encoding='utf-8')), args.output)


if __name__ == '__main__':
    main()
