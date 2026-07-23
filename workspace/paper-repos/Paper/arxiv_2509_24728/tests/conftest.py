"""Shared pytest fixtures for the catnat test suite."""

import pytest
import torch


@pytest.fixture(autouse=True)
def set_deterministic_seed():
    """Fix random seed before every test for reproducibility."""
    torch.manual_seed(42)
    yield


@pytest.fixture
def device():
    return torch.device("cpu")   # Tests always run on CPU


@pytest.fixture(params=[4, 8, 16])
def K_pow2(request):
    """Parametrised fixture: powers of 2 for K."""
    return request.param
