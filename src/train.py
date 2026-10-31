"""Compatibilita: implementazione in src.first_model.train."""
import sys
from .first_model import train as _implementation

if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
