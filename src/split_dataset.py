"""Crea una sola volta holdout 80/20 e fold stratificati del train."""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .dataset import number


def digest(content):
    return hashlib.sha256(content).hexdigest()


def csv_bytes(rows, columns):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def parse_rows(content):
    reader = csv.DictReader(io.StringIO(content.decode('utf-8-sig')))
    columns = reader.fieldnames
    if not columns or len(columns) != len(set(columns)):
        raise ValueError('Intestazioni CSV mancanti o duplicate')
    required = {'test_id', 'gender', 'age_years', 'P_TLac_W'}
    if not required <= set(columns):
        raise ValueError(f'Colonne richieste: {sorted(required)}')
    rows, seen = [], set()
    for source in reader:
        if None in source or any(value is None for value in source.values()):
            raise ValueError('Riga CSV malformata')
        row = dict(source)
        if not row['test_id'] or row['test_id'] in seen:
            raise ValueError('ID mancante o duplicato')
        seen.add(row['test_id'])
        if row['gender'] not in ('F', 'M'):
            raise ValueError('Sesso diverso da F/M')
        for key in set(columns) - {'test_id', 'gender'}:
            row[key] = number(source[key])
            if source[key].strip() and row[key] is None:
                raise ValueError(f'Numero non valido: {key}, {row["test_id"]}')
        if row['P_TLac_W'] is None or row['P_TLac_W'] <= 0:
            raise ValueError('Target non valido')
        if row['age_years'] is not None and row['age_years'] <= 0:
            raise ValueError('Età non valida')
        rows.append(row)
    if not rows:
        raise ValueError('Dataset vuoto')
    return rows, columns


def stratum(row):
    age = row['age_years']
    band = 'missing' if age is None else 'lt25' if age < 25 else '25to35' if age < 35 else 'ge35'
    return f'{row["gender"]}_{band}'


def make_folds(rows, n_folds, seed):
    """Round-robin stratificato: bilanciamento globale e per strato ±1."""
    if not isinstance(n_folds, int) or n_folds < 2 or len(rows) < 2 * n_folds:
        raise ValueError('Numero di fold incompatibile con la numerosità')
    rng = np.random.default_rng(seed)
    assignment, cursor = {}, 0
    labels = sorted({stratum(r) for r in rows})
    for label in labels:
        members = sorted(r['test_id'] for r in rows if stratum(r) == label)
        rng.shuffle(members)
        for test_id in members:
            assignment[test_id] = cursor % n_folds
            cursor += 1
    return [{'test_id': r['test_id'], 'fold': assignment[r['test_id']], 'stratum': stratum(r)} for r in rows]


def holdout_indices(labels, seed):
    """Quote proporzionali arrotondate, con almeno un record per parte/strato."""
    counts = Counter(labels)
    names = sorted(counts)
    total_test = int(np.ceil(.2 * len(labels)))
    if total_test < len(names) or len(labels) - total_test < len(names):
        raise ValueError('Numerosità insufficiente per rappresentare gli strati nello split 80/20')
    quota = {name: .2 * counts[name] for name in names}
    allocation = {name: min(counts[name] - 1, max(1, int(quota[name]))) for name in names}
    while sum(allocation.values()) < total_test:
        candidates = [name for name in names if allocation[name] < counts[name] - 1]
        selected = max(candidates, key=lambda name: quota[name] - allocation[name])
        allocation[selected] += 1
    while sum(allocation.values()) > total_test:
        candidates = [name for name in names if allocation[name] > 1]
        selected = max(candidates, key=lambda name: allocation[name] - quota[name])
        allocation[selected] -= 1
    rng = np.random.default_rng(seed)
    train, test = [], []
    for name in names:
        members = np.array([i for i, label in enumerate(labels) if label == name])
        rng.shuffle(members)
        test.extend(members[:allocation[name]])
        train.extend(members[allocation[name]:])
    return train, test


def create_split(config):
    if set(config) - {'source', 'output', 'seed', 'test_fraction', 'cv_folds'}:
        raise ValueError('Chiavi di configurazione split sconosciute')
    destination = Path(config['output'])
    if destination.exists():
        raise FileExistsError(f'Split già presente, nessuna sovrascrittura: {destination}')
    if config.get('test_fraction', .2) != .2:
        raise ValueError('Questo studio richiede split 80/20')
    content = Path(config['source']).read_bytes()
    rows, columns = parse_rows(content)
    rows.sort(key=lambda r: r['test_id'])
    labels = [stratum(r) for r in rows]
    counts = Counter(labels)
    if min(counts.values()) < 2:
        raise ValueError('Strato con meno di due test: impossibile separarlo in train/test')
    seed, n_folds = config.get('seed', 42), config.get('cv_folds', 5)
    train_idx, test_idx = holdout_indices(labels, seed)
    train = [rows[i] for i in sorted(train_idx)]
    test = [rows[i] for i in sorted(test_idx)]
    train_ids, test_ids = {r['test_id'] for r in train}, {r['test_id'] for r in test}
    if train_ids & test_ids or len(train_ids | test_ids) != len(rows):
        raise ValueError('Partizioni non disgiunte o incomplete')
    if {stratum(r) for r in train} != set(labels) or {stratum(r) for r in test} != set(labels):
        raise ValueError('Stratificazione non rappresentata in entrambe le partizioni')
    folds = make_folds(train, n_folds, seed)
    train_content = csv_bytes(train, columns)
    test_content = csv_bytes(test, columns)
    folds_content = csv_bytes(folds, ['test_id', 'fold', 'stratum'])
    train_manifest = {'train_sha256': digest(train_content), 'folds_sha256': digest(folds_content),
                      'n_train': len(train), 'cv_folds': n_folds, 'seed': seed,
                      'age_bands': ['<25', '25–<35', '≥35', 'missing'],
                      'small_strata': {k: v for k, v in Counter(stratum(r) for r in train).items() if v < n_folds}}
    manifest = {**train_manifest, 'source_sha256': digest(content), 'config': config,
                'n_test': len(test), 'test_sha256': digest(test_content),
                'strata': {k: {'total': counts[k],
                               'train': sum(stratum(r) == k for r in train),
                               'test': sum(stratum(r) == k for r in test)} for k in sorted(counts)},
                'unit': 'test ciclistico; atleta non identificabile',
                'holdout_policy': 'Non aprire test.csv dopo la creazione senza nuovo ordine esplicito'}
    # Tutte le verifiche e gli hash precedono la scrittura. Nessuna rilettura del test.
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'train.csv').write_bytes(train_content)
    (destination / 'cv_folds.csv').write_bytes(folds_content)
    (destination / 'train_manifest.json').write_text(json.dumps(train_manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (destination / 'split_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (destination / 'test.csv').write_bytes(test_content)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    result = create_split(json.loads(args.config.read_text(encoding='utf-8')))
    print(f'Split creato: {result["n_train"]} train, {result["n_test"]} test. Holdout da non aprire.')


if __name__ == '__main__':
    main()
