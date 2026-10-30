"""Lettura conservativa degli Excel: nessuna correzione o imputazione."""

import math
import warnings
from datetime import date, datetime
from pathlib import Path

import openpyxl


# key, label originale, unità, gruppo. Le unità ematologiche non sono dichiarate.
FIELDS = [
    ('height_m', 'Height', 'm', 'antropometria'),
    ('mass_kg', 'Weight', 'kg', 'antropometria'),
    ('bmi_kg_m2', 'BMI', 'kg/m²', 'antropometria'),
    ('skinfolds_mm', 'Sum of skinfolds', 'mm', 'antropometria'),
    ('fat_pct', 'Fat', '%', 'antropometria'),
    ('lbm_kg', 'LBM', 'kg', 'antropometria'),
    ('hb', 'Hb', 'non dichiarata', 'ematologia'),
    ('hct', 'Hct', 'non dichiarata', 'ematologia'),
    ('rbc', 'RBC', 'non dichiarata', 'ematologia'),
    ('wbc', 'WBC', 'non dichiarata', 'ematologia'),
    ('vo2max_ml_kg_min', 'VO2max', 'mL/kg/min', 'cardiorespiratorio'),
    ('vo2max_l_min', 'VO2max (absolute)', 'L/min', 'cardiorespiratorio'),
    ('ve_max_l_min', 'Vemax', 'L/min', 'cardiorespiratorio'),
    ('hr_max_bpm', 'Hrmax', 'bpm', 'cardiorespiratorio'),
    ('P_TLac_W', 'Load @ Tlac', 'W', 'lattato'),
    ('hr_tlac_bpm', 'HR @ Tlac', 'bpm', 'lattato'),
    ('vo2max_tlac_pct', '%VO2max @ Tlac', '%', 'lattato'),
    ('hrmax_tlac_pct', '%HRmax @ Tlac', '%', 'lattato'),
    ('bla_tlac_mmol_l', 'Bla @ Tlac', 'mmol/L', 'lattato'),
    ('power_2mmol_W', 'Load @ 2mmol', 'W', 'lattato'),
    ('hr_2mmol_bpm', 'HR @ 2mmol', 'bpm', 'lattato'),
    ('bla_160bpm_mmol_l', 'Bla @ 160 bpm', 'mmol/L', 'lattato'),
    ('vo2_160bpm_ml_kg_min', 'VO2 @ 160bpm', 'mL/kg/min', 'cardiorespiratorio'),
    ('power_160bpm_W', 'Load @ 160 bpm', 'W', 'cardiorespiratorio'),
]
SIGNALS = [
    ('power_W', 'Actual Power', 'W'), ('hr_bpm', 'HR', 'bpm'),
    ('vo2_ml_kg_min', 'VO2', 'mL/kg/min'), ('bla_mmol_l', 'Bla', 'mmol/L'),
    ('cadence_rpm', 'SR/Cadence', 'rpm'),
]


def number(value):
    """Solo numeri finiti o stringhe numeriche non ambigue."""
    if value is None or isinstance(value, (bool, datetime, date)):
        return None
    try:
        parsed = float(value)
    except (ValueError, TypeError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        for fmt in ('%d/%m/%Y', '%Y-%m-%d'):
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                pass
    return None


def serial(value):
    return value.isoformat() if isinstance(value, (date, datetime)) else value


def active_steps(steps):
    return [row for row in steps if row.get('power_W') is not None and row['power_W'] > 0]


def target_status(target, steps):
    if target is None:
        return 'mancante/non numerico'
    if target <= 0:
        return 'non positivo'
    powers = [r['power_W'] for r in active_steps(steps)]
    if not powers:
        return 'intervallo non disponibile'
    if not min(powers) <= target <= max(powers):
        return 'fuori intervallo osservato'
    return 'entro intervallo osservato'


def read_workbook(path):
    # Non salviamo mai i workbook: le estensioni non supportate restano negli originali.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unknown extension is not supported')
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheets = {s.title: [list(row) for row in s.values] for s in book}
        book.close()
    return sheets


def load_test(path):
    path = Path(path)
    sheets = read_workbook(path)
    if set(sheets) != {'Sheet1', 'Sheet2', 'Sheet3'}:
        raise ValueError(f'{path.name}: struttura fogli inattesa')
    original = {name: [[serial(v) for v in r] for r in rows] for name, rows in sheets.items()}
    scalar = {}
    issues = []
    for name in ('Sheet1', 'Sheet3'):
        for i, row in enumerate(sheets[name], 1):
            if row and row[0] is not None:
                scalar[str(row[0]).strip()] = row[1] if len(row) > 1 else None
            if any(v is not None and str(v).strip() for v in row[2:]):
                issues.append(f'{name}, riga {i}: celle aggiuntive conservate nella tabella')
    features = {}
    for key, label, unit, group in FIELDS:
        value = scalar.get(label)
        features[key] = number(value)
        if value is not None and features[key] is None:
            issues.append(f'{label}: valore non numerico {value!r}')
    birth = parse_date(scalar.get('DOB:'))
    tested = parse_date(scalar.get('Test Date:'))
    features['age_years'] = (tested - birth).days / 365.2425 if birth and tested else None
    for label, parsed in [('DOB:', birth), ('Test Date:', tested)]:
        if parsed is None:
            issues.append(f'{label} data mancante/non interpretabile')
    mass = features['mass_kg']
    target = features['P_TLac_W']
    features['P_TLac_W_kg'] = target / mass if mass and mass > 0 and target is not None and target > 0 else None
    source = sheets['Sheet2']
    expected = [s[1] for s in SIGNALS]
    if source[0][:5] != expected:
        raise ValueError(f'{path.name}: intestazioni segnali inattese')
    steps = []
    for i, row in enumerate(source[1:], 2):
        if not any(v is not None and str(v).strip() for v in row):
            continue
        step = {}
        for j, (key, label, unit) in enumerate(SIGNALS):
            raw = row[j] if j < len(row) else None
            step[key] = number(raw)
            if raw is not None and step[key] is None:
                issues.append(f'Sheet2 riga {i}, {label}: valore non numerico {raw!r}')
        if any(v is not None and str(v).strip() for v in row[5:]):
            issues.append(f'Sheet2 riga {i}: celle aggiuntive conservate nella tabella')
        steps.append(step)
    active = active_steps(steps)
    features['active_steps_n'] = len(active)
    features['max_power_W'] = max((r['power_W'] for r in active), default=None)
    for key, label, unit in SIGNALS[1:]:
        values = sorted(r[key] for r in active if r[key] is not None)
        n = len(values)
        features[f'median_{key}'] = (values[(n - 1) // 2] + values[n // 2]) / 2 if n else None
    status = target_status(target, steps)
    if status != 'entro intervallo osservato':
        issues.append(f'T_Lac: {status}')
    powers = [r['power_W'] for r in active]
    if any(b <= a for a, b in zip(powers, powers[1:])):
        issues.append('Potenza attiva non strettamente crescente')
    for step in steps:
        if step['power_W'] is None or step['power_W'] < 0:
            issues.append('Step con potenza mancante/non valida: escluso dai grafici')
        for key in ('hr_bpm', 'vo2_ml_kg_min', 'bla_mmol_l', 'cadence_rpm'):
            if step[key] is not None and step[key] < 0:
                issues.append(f'{key}: valore negativo conservato')
    return {
        'id': path.stem, 'file': path.name,
        'gender': str(scalar.get('Gender:', '')).strip(),
        'test_date': tested.isoformat() if tested else None,
        'features': features, 'steps': steps,
        'personal_raw': original['Sheet1'], 'signals_raw': original['Sheet2'],
        'derived_raw': original['Sheet3'], 'issues': issues, 'target_status': status,
    }


def load_dataset(root):
    root = Path(root)
    summary = next(iter(read_workbook(root / 'Data_Summary.xlsx').values()))
    header = summary[0]
    index = [dict(zip(header, row)) for row in summary[1:] if any(v is not None for v in row)]
    files = [r['file'] for r in index]
    if len(files) != len(set(files)):
        raise ValueError('File duplicati nell’indice')
    if set(files) != {p.name for p in root.glob('A*.xlsx')}:
        raise ValueError('Indice e file estratti non coincidono')
    records = []
    for row in sorted((r for r in index if r['sport'] == 'Cycling'), key=lambda r: int(r['file'][1:-5])):
        record = load_test(root / row['file'])
        record['summary_raw'] = {str(k): serial(v) for k, v in row.items() if k is not None}
        for summary_key, key in [('lt_load', 'P_TLac_W'), ('vo2max', 'vo2max_ml_kg_min'),
                                 ('HRmax', 'hr_max_bpm'), ('height', 'height_m'), ('weight', 'mass_kg')]:
            a, b = number(row.get(summary_key)), record['features'][key]
            if (a is None) != (b is None) or (a is not None and not math.isclose(a, b, abs_tol=1e-8)):
                record['issues'].append(f'Indice {summary_key}={row.get(summary_key)!r} diverso dal test ({b!r}); EDA usa il test')
        records.append(record)
    return records, index
