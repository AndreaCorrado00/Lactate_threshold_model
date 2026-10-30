"""Dataset descrittivo da manifest EDA; nessun addestramento predittivo."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .analysis import feature_metadata
from .dataset import number


FEATURES = [f['key'] for f in feature_metadata()
            if f['group'] != 'lattato' and f['key'] != 'median_cadence_rpm']
SETS = {'A': [k for k in FEATURES if 'vo2' not in k], 'B': FEATURES}
TARGET = 'P_TLac_W'


def select_records(records):
    rows, excluded, seen = [], [], set()
    for record in records:
        test_id = record['id']
        if test_id in seen:
            raise ValueError(f'ID duplicato: {test_id}')
        seen.add(test_id)
        target = number(record['features'].get(TARGET))
        if target is None or target <= 0:
            excluded.append({'test_id': test_id, 'reason': 'target mancante/non finito' if target is None else 'target non positivo'})
            continue
        rows.append({'test_id': test_id, 'gender': record['gender'].strip().upper(),
                     TARGET: target,
                     **{key: number(record['features'].get(key)) for key in FEATURES}})
    if not rows:
        raise ValueError('Nessun test con target valido: arresto all’audit')
    return rows, excluded


class SexMedianImputer:
    """Fit separato da transform per consentire il riuso sul solo train."""

    def __init__(self, keys):
        self.keys = list(keys)
        self.medians = {}

    def fit(self, rows):
        medians = {}
        for gender in sorted({r['gender'] for r in rows}):
            if gender not in ('F', 'M'):
                raise ValueError(f'sesso non supportato: {gender!r}')
            medians[gender] = {}
            for key in self.keys:
                values = [number(r.get(key)) for r in rows if r['gender'] == gender]
                observed = [v for v in values if v is not None]
                if not observed:
                    raise ValueError(f'Impossibile stimare mediana: {gender}, {key}')
                medians[gender][key] = float(np.median(observed))
        if not medians:
            raise ValueError('Nessuna mediana disponibile')
        self.medians = medians
        return self

    def transform(self, rows):
        result = []
        for row in rows:
            if row['gender'] not in self.medians:
                raise ValueError(f'sesso assente dal fit: {row["gender"]!r}')
            result.append({**row, **{key: number(row.get(key)) if number(row.get(key)) is not None
                                    else self.medians[row['gender']][key] for key in self.keys}})
        return result


def vif_table(rows, keys):
    """VIF = 1/(1-R²), OLS ausiliarie con intercetta (centratura).

    Standardizzazione per stabilità numerica; costanti escluse dai regressori.
    Non si usa la pseudoinversa della matrice di correlazione per i VIF.
    """
    matrix = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError('VIF richiede una matrice completa e finita')
    centered = matrix - matrix.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    active = np.flatnonzero(norms > 0)
    result = []
    for j, key in enumerate(keys):
        value, status = None, 'costante'
        if j in active:
            if len(rows) <= len(active):
                status = 'numerosità insufficiente'
            else:
                y = centered[:, j] / norms[j]
                others = active[active != j]
                x = centered[:, others] / norms[others]
                residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
                unexplained = float(residual @ residual)
                value = float('inf') if unexplained < 1e-12 else max(1., 1. / unexplained)
                status = 'dipendenza esatta/quasi esatta' if np.isinf(value) else 'stimato'
        result.append({'feature': key, 'vif': value, 'status': status, 'n': len(rows)})
    return result


def write_csv(path, rows, columns):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build(manifest, output, report_dir):
    content = manifest.read_bytes()
    payload = json.loads(content)
    rows, excluded = select_records(payload['records'])
    complete = [r for r in rows if r['gender'] in ('F', 'M')
                and all(r[key] is not None for key in FEATURES)]
    encoded = [{**r, 'sex_M': float(r['gender'] == 'M')} for r in complete]
    vif = []
    for name, keys in SETS.items():
        table = vif_table(encoded, ['sex_M', *keys]) if encoded else [
            {'feature': key, 'vif': None, 'status': 'nessun caso completo', 'n': 0}
            for key in ['sex_M', *keys]]
        vif.extend({'set': name, **r} for r in table)
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    columns = ['test_id', 'gender', TARGET, *FEATURES]
    write_csv(output / 'features_unimputed.csv', rows, columns)
    write_csv(output / 'excluded_tests.csv', excluded, ['test_id', 'reason'])
    write_csv(report_dir / 'vif.csv', vif, ['set', 'feature', 'vif', 'status', 'n'])
    metadata = {'manifest': str(manifest), 'manifest_sha256': hashlib.sha256(content).hexdigest(),
                'source_archive_md5': payload.get('md5'), 'n_input': len(payload['records']),
                'n_retained': len(rows), 'n_excluded': len(excluded), 'target': TARGET,
                'feature_sets': SETS, 'features': [f for f in feature_metadata() if f['key'] in FEATURES],
                'imputation_scope': 'nessuna imputazione',
                'excluded_features': ['median_cadence_rpm'],
                'vif_scope': 'casi completi comuni ai set A/B, analisi esplorativa',
                'vif_n': len(complete)}
    (output / 'features_manifest.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    retained = {r['test_id'] for r in rows}
    flagged = [r['id'] for r in payload['records'] if r['id'] in retained
               and r.get('target_status') != 'entro intervallo osservato']
    lines = ['# Dataset delle feature e multicollinearità',
             f'{len(payload["records"])} test iniziali; {len(rows)} mantenuti; {len(excluded)} esclusi.',
             'Una riga per test ciclistico, non necessariamente per atleta distinto. Non esiste un ID atleta stabile.',
             '## Coorte e provenienza',
             f'Manifest: `{manifest}`; SHA256 `{metadata["manifest_sha256"]}`. '
             'Il JSON è l’estrazione prodotta dall’audit, non viene modificato. '
             'L’integrità dello ZIP è verificata da `python -m src.build_report`.',
             'Esclusioni: ' + ', '.join(f'{r["test_id"]} ({r["reason"]})' for r in excluded) + '.',
             'Target con segnalazioni conservati: ' + ', '.join(flagged) + '.',
             '## Colonne e imputazione',
             '`test_id` è metadato, `gender` è covariata categorica e `P_TLac_W` è il target in W. '
             'Tutti gli indici derivati dal lattato, incluso il target W/kg, sono esclusi dalle feature. '
             'Il target non viene imputato. Non si ricava W/kg dalla massa imputata.',
             'Il dataset conserva i mancanti: nessuna imputazione prima dello split train/test. '
             'La cadenza è esclusa dalle feature per la scarsa disponibilità di misure osservate. '
             'Valori osservati ed estremi sono conservati. Le unità ematologiche non sono dichiarate dalla fonte.',
             'A = tutte le covariate disponibili non derivate dal lattato, escluse le misure VO₂; '
             'B = A più le quattro misure VO₂. Entrambi includono sesso, antropometria, ematologia, '
             'ventilazione, frequenza cardiaca, potenza e riassunti del protocollo. '
             'Questa è una definizione operativa esplorativa, non una selezione validata.',
             'A: ' + ', '.join(['gender', *SETS['A']]),
             'B aggiunge: ' + ', '.join(k for k in SETS['B'] if k not in SETS['A']),
             '| Feature | Mancanti conservati | % coorte |', '| --- | ---: | ---: |']
    for key in FEATURES:
        missing = sum(r[key] is None for r in rows)
        lines.append(f'| {key} | {missing} | {100 * missing / len(rows):.1f} |')
    lines += ['La cadenza resta disponibile soltanto nel manifest originale dell’EDA.',
              '## VIF',
              'Regressione OLS di ciascuna feature sulle altre con intercetta: VIF = 1/(1−R²). '
              'Sesso codificato F=0/M=1. Target e ID esclusi. Calcolo separato per A e B sulle stesse righe. '
              f'Si usano {len(complete)} casi completi comuni ai due set, senza imputazione; '
              f'{len(rows) - len(complete)} righe escluse soltanto dal VIF restano nel CSV. '
              'Il sottocampione completo può non rappresentare l’intera coorte. '
              'Costanti segnalate e rimosse dai regressori ausiliari; dipendenze esatte/quasi esatte riportate come inf. '
              'Con n ≤ numero di colonne non costanti il VIF è dichiarato non stimabile.',
              '| Set | Feature | VIF | Stato |', '| --- | --- | ---: | --- |']
    for r in sorted(vif, key=lambda r: (r['set'], -(r['vif'] or 0))):
        value = 'NA' if r['vif'] is None else f'{r["vif"]:.2f}'
        lines.append(f'| {r["set"]} | {r["feature"]} | {value} | {r["status"]} |')
    lines += ['## Interpretazione e prossimi passi',
              'Nel set B i VIF maggiori riguardano: ' + ', '.join(
                  f'{r["feature"]} ({r["vif"]:.2f})'
                  for r in sorted((r for r in vif if r['set'] == 'B' and r['vif'] is not None),
                                  key=lambda r: r['vif'], reverse=True)[:5]) + '.',
              'VIF elevati indicano instabilità potenziale dei coefficienti lineari condizionata alle altre '
              'feature, non assenza di valore predittivo. Nessuna colonna è eliminata automaticamente. '
              'Prima di ridurre le feature, confrontare rappresentazioni alternative di massa/altezza/BMI, '
              'massa/massa magra/grasso e VO₂ assoluto/relativo, e la ridondanza dei riassunti del protocollo. '
              'Le relazioni fisiologiche non lineari non sono tutte rilevate dal VIF.',
              '**Il VIF sui casi completi è esplorativo.** Per valutare modelli usare '
              '`features_unimputed.csv`, creare lo split 80/20 e stimare mediane, scaling, VIF e selezione solo '
              'sul train (e sui train fold durante il tuning). `SexMedianImputer` separa fit e transform. '
              'Non sono stati creati split né addestrati modelli. Il rischio di test ripetuti resta aperto.',
              'Studio retrospettivo/esplorativo: nessuna validità clinica attribuita. '
              'T_Lac non è equiparata a IAT, LT1, LT2 o MLSS.']
    markdown = ''
    previous_table = False
    for line in lines:
        is_table = line.startswith('|')
        markdown += ('\n' if is_table and previous_table else '\n\n') + line
        previous_table = is_table
    (report_dir / 'features.md').write_text(markdown.strip() + '\n', encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/processed/cycling.json'))
    parser.add_argument('--output', type=Path, default=Path('data/processed'))
    parser.add_argument('--reports', type=Path, default=Path('reports'))
    args = parser.parse_args()
    result = build(args.manifest, args.output, args.reports)
    print(f'Test mantenuti: {result["n_retained"]}; esclusi: {result["n_excluded"]}')


if __name__ == '__main__':
    main()
