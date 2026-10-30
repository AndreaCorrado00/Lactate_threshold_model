import pytest
from src.analysis import correlation_matrix, flag_extremes


def test_pairwise_missing_and_spearman_ties():
    rows = [{'x': 1, 'y': 10}, {'x': 2, 'y': 20}, {'x': 3, 'y': 20},
            {'x': 4, 'y': None}, {'x': None, 'y': 100}]
    result = correlation_matrix(rows, ['x', 'y'], 'spearman')
    assert result['n'][0][1] == 3
    assert result['z'][0][1] == pytest.approx(.8660254038)


def test_constant_or_insufficient_pairs_are_undefined():
    rows = [{'x': 1, 'y': 2}, {'x': 1, 'y': 3}]
    result = correlation_matrix(rows, ['x', 'y'], 'pearson')
    assert result['z'][0][1] is None
    assert result['n'][0][1] == 2


def test_extreme_values_are_flagged_without_replacing_them():
    rows = [{'features': {'mass_kg': x}, 'issues': []} for x in [60, 65, 70, 75, 80, 500]]
    flag_extremes(rows)
    assert rows[-1]['features']['mass_kg'] == 500
    assert any('mass_kg' in issue for issue in rows[-1]['issues'])
    assert not rows[2]['issues']
