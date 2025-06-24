"""Test doubles that must be importable so joblib can unpickle them inside the service."""
import numpy as np


class FaultyModel:
    """Registered as a bad candidate: every inference raises."""

    def predict_proba(self, x):
        raise RuntimeError("boom")


class ConstantModel:
    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, x):
        n = len(x)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])
