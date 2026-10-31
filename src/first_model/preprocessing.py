"""Preprocessing persistibile, stimato esclusivamente sui dati di fit."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..build_features import SexMedianImputer


class FoldPreprocessor:
    """Mediane per sesso del solo fold train, con scaling opzionale."""

    def __init__(self, features, scale=False, pca=None):
        self.features = list(features)
        self.scale = scale
        self.pca_config = pca
        if pca is not None:
            if not isinstance(pca, dict) or set(pca) != {'n_components'} or not scale:
                raise ValueError('PCA richiede scaling=true e solo n_components')
            n = pca['n_components']
            if not (type(n) is int and n >= 1 or type(n) is float and 0 < n < 1):
                raise ValueError('n_components: intero positivo o quota di varianza fra 0 e 1')
            if not any(f != 'gender' for f in features):
                raise ValueError('PCA richiede covariate numeriche')

    def matrix(self, rows):
        filled = self.imputer.transform(rows)
        return np.array([[float(r['gender'] == 'M') if key == 'gender' else r[key]
                          for key in self.features] for r in filled], dtype=float)

    def fit(self, rows):
        self.imputer = SexMedianImputer([k for k in self.features if k != 'gender']).fit(rows)
        if self.pca_config is not None:
            self.numeric_indices = [i for i, f in enumerate(self.features) if f != 'gender']
            values = self.matrix(rows)[:, self.numeric_indices]
            self.scaler = StandardScaler().fit(values)
            standardized = self.scaler.transform(values)
            if not np.any(np.var(standardized, axis=0) > 0):
                raise ValueError('PCA non definita: tutte le covariate sono costanti')
            self.pca_ = PCA(n_components=self.pca_config['n_components'], svd_solver='full').fit(standardized)
            return self
        self.scaler = StandardScaler().fit(self.matrix(rows)) if self.scale else None
        return self

    def transform(self, rows):
        values = self.matrix(rows)
        if getattr(self, 'pca_config', None) is not None:
            transformed = self.pca_.transform(self.scaler.transform(values[:, self.numeric_indices]))
            if 'gender' in self.features:
                transformed = np.column_stack([transformed, values[:, self.features.index('gender')]])
            return transformed
        return self.scaler.transform(values) if self.scaler is not None else values

    def pca_metadata(self):
        if getattr(self, 'pca_config', None) is None:
            return None
        return {'numeric_features': [self.features[i] for i in self.numeric_indices],
                'n_components': int(self.pca_.n_components_),
                'components': self.pca_.components_.tolist(),
                'explained_variance_ratio': self.pca_.explained_variance_ratio_.tolist(),
                'explained_variance': self.pca_.explained_variance_.tolist(),
                'gender_passthrough': 'gender' in self.features}
