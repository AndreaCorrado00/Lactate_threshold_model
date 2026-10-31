"""Predittore composto: offset proporzionale e correzione multivariata."""

import numpy as np

from ..first_model.preprocessing import FoldPreprocessor
from ..first_model.train import make_model


def maximum_power(rows):
    try:
        values = np.asarray([row['max_power_W'] for row in rows], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('max_power_W richiesta, numerica e positiva') from error
    if not len(values) or not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError('max_power_W richiesta, finita e positiva; non viene imputata')
    return values


def validate_proportional(config):
    if not isinstance(config, dict):
        raise ValueError('proportional deve essere un oggetto')
    if config.get('mode') == 'fit' and set(config) == {'mode'}:
        return
    if config.get('mode') == 'fixed' and set(config) == {'mode', 'k'}:
        value = config['k']
        if type(value) in (int, float) and np.isfinite(value) and value > 0:
            return
    raise ValueError('proportional: mode=fit oppure mode=fixed con k finito positivo')


def estimate_k(rows, proportional):
    validate_proportional(proportional)
    x = maximum_power(rows)
    if proportional['mode'] == 'fixed':
        return float(proportional['k'])
    y = np.asarray([row['P_TLac_W'] for row in rows], dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError('Target non finito nel fit di k')
    k = float(np.dot(x, y) / np.dot(x, x))
    if not np.isfinite(k):
        raise ValueError('Coefficiente proporzionale non finito')
    return k


class OffsetRegressor:
    """Il fit riceve solo il training locale; predict non usa mai il target."""

    def __init__(self, features, model_name, params, scaling=True, seed=42,
                 target_mode='residual', proportional=None):
        if target_mode not in ('residual', 'direct'):
            raise ValueError('target_mode deve essere residual o direct')
        if type(scaling) is not bool:
            raise ValueError('scaling deve essere booleano')
        self.features = list(features)
        if not self.features or 'max_power_W' in self.features:
            raise ValueError('max_power_W non ammessa nelle feature di correzione')
        self.model_name = model_name
        self.params = dict(params)
        self.scaling = scaling
        self.seed = seed
        self.target_mode = target_mode
        self.proportional = {'mode': 'fit'} if proportional is None else dict(proportional)
        validate_proportional(self.proportional)

    def fit(self, rows):
        self.k_ = estimate_k(rows, self.proportional)
        y = np.asarray([row['P_TLac_W'] for row in rows], dtype=float)
        target = y - self.k_ * maximum_power(rows) if self.target_mode == 'residual' else y
        self.preprocessor_ = FoldPreprocessor(self.features, scale=self.scaling).fit(rows)
        self.estimator_ = make_model(self.model_name, self.params, self.seed)
        self.estimator_.fit(self.preprocessor_.transform(rows), target)
        return self

    def predict_parts(self, rows):
        baseline = self.k_ * maximum_power(rows)
        raw_prediction = np.asarray(self.estimator_.predict(self.preprocessor_.transform(rows)), dtype=float)
        if raw_prediction.shape != baseline.shape or not np.all(np.isfinite(raw_prediction)):
            raise ValueError('Predizioni non finite o di dimensione errata')
        if self.target_mode == 'residual':
            delta = raw_prediction
            prediction = baseline + delta
        else:
            prediction = raw_prediction
            delta = prediction - baseline
        return baseline, delta, prediction

    def predict(self, rows):
        return self.predict_parts(rows)[2]

    def metadata(self):
        prep = self.preprocessor_
        return {'k': self.k_, 'features': self.features, 'medians': prep.imputer.medians,
                'scaler_mean': prep.scaler.mean_.tolist() if prep.scaler else None,
                'scaler_scale': prep.scaler.scale_.tolist() if prep.scaler else None}
