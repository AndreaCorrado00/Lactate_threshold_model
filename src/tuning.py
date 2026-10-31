"""Compatibilita: implementazione in src.first_model.tuning."""
import sys
from .first_model import tuning as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
