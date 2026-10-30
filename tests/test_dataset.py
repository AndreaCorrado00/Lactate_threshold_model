from datetime import datetime

import openpyxl
import pytest

from src.dataset import active_steps, load_test, number, target_status


def test_numeric_missing_and_malformed_are_not_zero():
    assert number(None) is None
    assert number('-') is None
    assert number('45/7') is None
    assert number('7.7.') is None
    assert number(0) == 0
    assert number(' 12.5 ') == 12.5
    assert number(float('inf')) is None


def test_rest_is_retained_in_source_but_excluded_from_active_steps():
    rows = [{'power_W': 0, 'hr_bpm': 80}, {'power_W': 120, 'hr_bpm': 135},
            {'power_W': None, 'hr_bpm': 150}, {'power_W': -1, 'hr_bpm': 90}]
    assert active_steps(rows) == [rows[1]]
    assert len(rows) == 4


def test_target_status_does_not_equate_presence_with_validity():
    steps = [{'power_W': 120}, {'power_W': 200}]
    assert target_status(None, steps) == 'mancante/non numerico'
    assert target_status(0, steps) == 'non positivo'
    assert target_status(210, steps) == 'fuori intervallo osservato'
    assert target_status(164, steps) == 'entro intervallo osservato'


def test_workbook_extracts_units_raw_missing_and_unexpected_cells(tmp_path):
    path = tmp_path / 'A999.xlsx'
    w = openpyxl.Workbook()
    s = w.active
    s.title = 'Sheet1'
    for row in [('Name:', None), ('Gender:', 'F'), ('Test:', 'Cycling'),
                ('Test Date:', '23/02/2005'), ('DOB:', datetime(1971, 10, 9)),
                ('Height', 1.67), ('Weight', 51.3)]:
        s.append(row)
    s = w.create_sheet('Sheet2')
    s.append(['Actual Power', 'HR', 'VO2', 'Bla', 'SR/Cadence'])
    s.append([0, 80, '7.7.', 0.6, None])
    s.append([120, 135, 36.5, 1, None, 'nota originale'])
    s.append([200, 180, 52.2, 5.9, None])
    s = w.create_sheet('Sheet3')
    s.append(['Load @ Tlac', 164])
    s.append(['VO2max', '45/7'])
    s.append(['Hrmax'])
    w.save(path)
    record = load_test(path)
    assert record['features']['mass_kg'] == 51.3
    assert record['features']['P_TLac_W'] == 164
    assert record['features']['P_TLac_W_kg'] == pytest.approx(164 / 51.3)
    assert record['features']['vo2max_ml_kg_min'] is None
    assert record['steps'][0]['vo2_ml_kg_min'] is None
    assert record['signals_raw'][1][2] == '7.7.'
    assert record['signals_raw'][2][5] == 'nota originale'
    assert record['features']['age_years'] == pytest.approx(33.3778, abs=.01)
    assert len(record['issues']) >= 2
