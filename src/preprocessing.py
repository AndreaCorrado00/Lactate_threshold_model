"""Compatibilita: implementazione in src.first_model.preprocessing."""
import sys
from .first_model import preprocessing as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
