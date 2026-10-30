import numpy as np
import pytest

from src.build_features import select_records, SexMedianImputer, vif_table


def test_target_filter_and_no_lactate_inputs():
    records = [dict(id=str(i), gender='M', features={'P_TLac_W': y})
               for i, y in enumerate([None, 0, -1, float('nan'), 200])]
    rows, excluded = select_records(records)
    assert [r['test_id'] for r in rows] == ['4']
    assert len(excluded) == 4
    assert 'P_TLac_W_kg' not in rows[0]
    assert 'median_bla_mmol_l' not in rows[0]
    with pytest.raises(ValueError, match='duplicat'):
        select_records([records[-1], records[-1]])


def test_sex_medians_fit_on_training_only():
    train = [dict(gender=g, x=x) for g, x in [('F', 2), ('F', 4), ('M', 10), ('M', 20)]]
    imputer = SexMedianImputer(['x']).fit(train)
    held_out = [dict(gender='F', x=None), dict(gender='M', x=None), dict(gender='F', x=1000)]
    assert [r['x'] for r in imputer.transform(held_out)] == [3, 15, 1000]
    assert held_out[0]['x'] is None
    with pytest.raises(ValueError, match='mediana'):
        SexMedianImputer(['x']).fit([dict(gender='F', x=None)])
    with pytest.raises(ValueError, match='sesso'):
        imputer.transform([dict(gender='', x=None)])


def test_vif_known_correlation_constants_and_exact_dependence():
    x = np.array([-1, -1, 1, 1.])
    z = np.array([-1, 1, -1, 1.])
    y = .5 * x + np.sqrt(.75) * z
    rows = [dict(x=a, y=b, constant=4) for a, b in zip(x, y)]
    table = {r['feature']: r for r in vif_table(rows, ['x', 'y', 'constant'])}
    assert table['x']['vif'] == pytest.approx(4 / 3)
    assert table['constant']['status'] == 'costante'
    for row in rows:
        row['duplicate'] = 2 * row['x']
    table = {r['feature']: r for r in vif_table(rows, ['x', 'duplicate'])}
    assert np.isinf(table['x']['vif'])


def test_vif_insufficient_rows():
    table = vif_table([dict(x=1, y=2), dict(x=2, y=3)], ['x', 'y'])
    assert all(r['status'] == 'numerosità insufficiente' for r in table)


def test_build_exports_consistent_tables_and_preserves_manifest(tmp_path):
    import csv
    import json
    from src.build_features import build, FEATURES

    records = [dict(id=f'A{i}', gender='F' if i % 2 else 'M',
                    target_status='entro intervallo osservato',
                    features={**{key: float(i + 1) for key in FEATURES}, 'P_TLac_W': 200})
               for i in range(30)]
    records[0]['features']['mass_kg'] = None
    records[-1]['features']['P_TLac_W'] = None
    manifest = tmp_path / 'source.json'
    content = json.dumps({'records': records})
    manifest.write_text(content, encoding='utf-8')
    metadata = build(manifest, tmp_path / 'out', tmp_path / 'reports')
    assert metadata['n_retained'] == 29
    assert manifest.read_text(encoding='utf-8') == content
    def read(name):
        with (tmp_path / 'out' / name).open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))
    raw = read('features_unimputed.csv')
    assert raw[0]['mass_kg'] == ''
    assert len(raw) == 29
    assert 'median_cadence_rpm' not in raw[0]
    assert not (tmp_path / 'out' / 'features.csv').exists()
    assert not (tmp_path / 'out' / 'imputation.csv').exists()
    assert 'medians' not in metadata
    with (tmp_path / 'reports' / 'vif.csv').open(encoding='utf-8') as stream:
        vif = list(csv.DictReader(stream))
    assert all(int(r['n']) == 28 for r in vif)
    assert all(r['feature'] != 'median_cadence_rpm' for r in vif)
