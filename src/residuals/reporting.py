"""Metriche e grafici del confronto appaiato con la proporzionalità."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..first_model.train import metrics


def safe_metrics(observed, predicted):
    result = metrics(observed, predicted)
    if np.ptp(observed) == 0:
        result['r2'] = None
    return result


def comparison_metrics(observed, baseline, predicted):
    observed, baseline, predicted = [np.asarray(v, dtype=float) for v in (observed, baseline, predicted)]
    delta = observed - baseline
    correction = predicted - baseline
    base = safe_metrics(observed, baseline)
    corrected = safe_metrics(observed, predicted)
    denominator = float(np.dot(delta, delta))
    q = 1 - float(np.sum((observed - predicted) ** 2)) / denominator if denominator > 0 else None
    correlation = (float(np.corrcoef(delta, correction)[0, 1])
                   if np.std(delta) > 0 and np.std(correction) > 0 else None)
    return {'proportional': base, 'corrected': corrected,
            'delta_zero': safe_metrics(delta, np.zeros_like(delta)),
            'delta_model': safe_metrics(delta, correction), 'q_delta': q,
            'delta_correlation': correlation,
            'mae_gain_W': base['mae_W'] - corrected['mae_W'],
            'rmse_gain_W': base['rmse_W'] - corrected['rmse_W'],
            'bias_W': float(np.mean(predicted - observed)),
            'observed_delta_sd_W': float(np.std(delta, ddof=1)),
            'predicted_delta_sd_W': float(np.std(correction, ddof=1))}


def write_report(output, summary, records):
    y = np.array([r['observed_W'] for r in records])
    prediction = np.array([r['predicted_W'] for r in records])
    baseline = np.array([r['baseline_W'] for r in records])
    delta, correction = y - baseline, prediction - baseline
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Potenza OOF', 'Scostamenti OOF dal riferimento del fold'])
    for values, label in [(baseline, 'Proporzionale'), (prediction, 'Modello')]:
        fig.add_trace(go.Scatter(x=y, y=values, mode='markers', name=label), row=1, col=1)
    fig.add_trace(go.Scatter(x=delta, y=correction, mode='markers', name='Correzione'), row=1, col=2)
    for column, obs, pred in [(1, y, prediction), (2, delta, correction)]:
        bounds = [float(min(obs.min(), pred.min())), float(max(obs.max(), pred.max()))]
        fig.add_trace(go.Scatter(x=bounds, y=bounds, mode='lines', name='Identita', showlegend=False), row=1, col=column)
        fig.update_xaxes(type='linear', title_text='Osservato (W)', row=1, col=column)
        fig.update_yaxes(type='linear', title_text='Predetto (W)', row=1, col=column)
    fig.update_layout(title=f"Confronto OOF: {summary['model']} / {summary['target_mode']}")
    fig.write_html(output / 'validation.html', include_plotlyjs=True)
    m = summary['oof']
    lines = ['# Proporzionale e correzione degli scostamenti', '',
             f"Modello: {summary['model']}; modalita: {summary['target_mode']}; {summary['n_oof']} test OOF.", '',
             '| Procedura | MAE (W) | RMSE (W) | R2 |', '|---|---:|---:|---:|']
    for key, label in [('proportional', 'Proporzionale'), ('corrected', 'Modello')]:
        v = m[key]
        lines.append(f"| {label} | {v['mae_W']:.3f} | {v['rmse_W']:.3f} | {v['r2']} |")
    lines += ['', f"Q_delta: {m['q_delta']}; correlazione delta: {m['delta_correlation']}.",
              f"Guadagno MAE: {m['mae_gain_W']:.3f} W; guadagno RMSE: {m['rmse_gain_W']:.3f} W. Positivo = migliore del proporzionale.", '',
              'k e preprocessing stimati solo sul training di ogni fold, anche interno, salvo modalita k fisso esplicita.',
              'La delta di ogni validation usa il k del suo training: nessun target residualizzato globalmente.',
              'Q_delta confronta la SSE con la correzione nulla; non coincide con R2 rispetto alla media di delta.',
              'La correzione Ridge include normalmente una intercetta: parte del guadagno puo essere una ricalibrazione media, non individualizzazione.',
              'Nessun accesso al test. Confronto esplorativo dopo sviluppo precedente: non conferma indipendente o effetto causale.',
              'Le righe identificano test, non necessariamente atleti indipendenti. Nessuna validita clinica attribuita.',
              'Il refit, se richiesto, ripete la selezione degli iperparametri sul solo train completo.']
    if summary['proportional']['mode'] == 'fixed':
        lines += ['', 'ATTENZIONE: k fisso; se derivato da tutta la coorte, la CV non isola la stima del riferimento.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
