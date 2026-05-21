"""quantum_smote package

Lightweight package initializer that exposes `QuantumSMOTE` when available.
"""
from importlib import import_module
from typing import Optional

__all__ = ["QuantumSMOTE", "__version__"]

__version__ = "0.1.0"


def _load_quantum_smote():
    try:
        mod = import_module("quantum_smote.smote.quantum_smote")
        return getattr(mod, "QuantumSMOTE")
    except Exception:
        return None


QuantumSMOTE = _load_quantum_smote()
