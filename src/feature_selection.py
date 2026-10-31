"""Compatibilita: implementazione in src.first_model.feature_selection."""
import sys
from .first_model import feature_selection as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
