"""Compatibilita: implementazione in src.first_model.proportional_baseline."""
import sys
from .first_model import proportional_baseline as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
