"""Compatibilita: implementazione in src.first_model.pca_report."""
import sys
from .first_model import pca_report as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
