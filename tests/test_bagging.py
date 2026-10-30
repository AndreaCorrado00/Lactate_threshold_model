import json

import numpy as np
import pytest

from src.train import effective_params, make_model
from src.tuning import sample_candidates


def test_bagging_nested_parameters_seed_and_serialization():
    params = {'n_estimators': 3, 'estimator__max_depth': 2,
              'estimator__min_samples_leaf': 2, 'max_samples': .8}
    model = make_model('bagging', params, 42)
    assert model.estimator.max_depth == 2
    assert model.estimator.min_samples_leaf == 2
    assert model.estimator.random_state == model.random_state == 42
    assert model.n_jobs == 1
    encoded = effective_params('bagging', params, 42)
    assert json.loads(json.dumps(encoded, allow_nan=False))['estimator']['params']['max_depth'] == 2
    x = np.arange(40).reshape(20, 2)
    y = x[:, 0] * 2
    np.testing.assert_allclose(model.fit(x, y).predict(x),
                               make_model('bagging', params, 42).fit(x, y).predict(x))


@pytest.mark.parametrize('params', [{'estimator__random_state': 1}, {'estimator': None},
                                   {'estimator__unknown': 3}])
def test_bagging_disallows_uncontrolled_estimator_or_seed(params):
    with pytest.raises(ValueError):
        make_model('bagging', params, 42)


def test_bagging_tuning_deduplicates_base_estimator_defaults():
    config = {'method': 'random', 'n_iter': 10,
              'param_spaces': {'bagging': {'estimator__max_depth': [None, 2]}}}
    candidates = sample_candidates({}, config, 'bagging', 42,
                                   canonicalize=lambda p: effective_params('bagging', p, 42))
    assert len(candidates) == 2
