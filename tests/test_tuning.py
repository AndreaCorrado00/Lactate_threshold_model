import pytest

from src.tuning import sample_candidates, search_parameters


def test_sampling_reproducible_unique_and_includes_base():
    base = {'alpha': 1.0}
    config = {'method': 'random', 'n_iter': 3, 'inner_folds': 3,
              'param_spaces': {'ridge': {'alpha': [.01, .1, 10., 100.]}}}
    candidates = sample_candidates(base, config, 'ridge', 42)
    assert candidates == sample_candidates(base, config, 'ridge', 42)
    assert candidates[0] == base
    assert len(candidates) == len({r['alpha'] for r in candidates}) == 3
    config['n_iter'] = 100
    assert len(sample_candidates(base, config, 'ridge', 42)) == 5


def test_sampling_deduplicates_implicit_model_defaults():
    config = {'method': 'random', 'n_iter': 20,
              'param_spaces': {'ridge': {'alpha': [1., 10., 10.]}}}
    candidates = sample_candidates({}, config, 'ridge', 42,
                                   canonicalize=lambda p: {'alpha': p.get('alpha', 1.)})
    assert candidates == [{}, {'alpha': 10.}]


def test_sampling_normalizes_real_parameters_but_preserves_count_fraction_semantics():
    config = {'method': 'random', 'n_iter': 20,
              'param_spaces': {'ridge': {'alpha': [1, 1., 10]}}}
    assert len(sample_candidates({'alpha': 1}, config, 'ridge', 42)) == 2
    config['param_spaces'] = {'random_forest': {'max_features': [1, 1.0]}}
    assert len(sample_candidates({'max_features': 1}, config, 'random_forest', 42)) == 2


@pytest.mark.parametrize('space', [{'alpha': []}, {'alpha': [float('nan')]}, {'alpha': 'bad'}])
def test_invalid_spaces_are_rejected(space):
    with pytest.raises(ValueError):
        sample_candidates({'alpha': 1.}, {'method': 'random', 'n_iter': 2,
                          'param_spaces': {'ridge': space}}, 'ridge', 42)


def test_search_uses_inner_score_and_repeats_selection(monkeypatch):
    import src.tuning as module
    calls = []
    rows = [{'test_id': 'train_only'}]
    def fake_select(received, features, factory, scaling, seed, inner_folds, improvement):
        assert received is rows
        params = factory()
        calls.append((params, seed, inner_folds))
        selected = ['mass_kg'] if params['depth'] == 2 else ['height_m']
        score = 4. if params['depth'] == 2 else 7.
        return selected, {'inner_mae_W': score, 'selected_features': selected}
    monkeypatch.setattr(module, 'forward_select', fake_select)
    candidates = [{'depth': 3}, {'depth': 2}]
    params, selected, audit = search_parameters(
        rows, ['mass_kg', 'height_m'], candidates, lambda p: p,
        False, 42, 3, {'min_improvement_W': .1})
    assert params == {'depth': 2}
    assert selected == ['mass_kg']
    assert audit['best_trial'] == 1
    assert len(calls) == 2
    assert all(call[1:] == (42, 3) for call in calls)
