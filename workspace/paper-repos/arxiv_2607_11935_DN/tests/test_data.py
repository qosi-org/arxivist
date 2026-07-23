"""Unit tests for ews_kalman.data (AIRS regional data loader)."""

from __future__ import annotations

import numpy as np
import pytest

from ews_kalman.data import AIRSDataLoader, N_OBSERVATIONS


def test_region_bounds_known_regions():
    loader = AIRSDataLoader()
    for region in ("arctic", "tropics", "monsoon"):
        bounds = loader.region_bounds(region)
        assert "lat_min" in bounds and "lat_max" in bounds


def test_region_bounds_unknown_region_raises():
    loader = AIRSDataLoader()
    with pytest.raises(ValueError):
        loader.region_bounds("antarctica")


def test_load_region_synthetic_fallback_shapes():
    loader = AIRSDataLoader()
    region = loader.load_region("arctic", data_dir="/nonexistent/path", seed=0)
    assert region["T"].shape == (N_OBSERVATIONS,)
    assert region["q"].shape == (N_OBSERVATIONS,)
    assert len(region["dates"]) == N_OBSERVATIONS


def test_load_region_no_fallback_raises_when_missing():
    loader = AIRSDataLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_region("arctic", data_dir="/nonexistent/path", use_synthetic_fallback=False)


def test_synthetic_fallback_is_deterministic_across_calls():
    loader = AIRSDataLoader()
    r1 = loader.load_region("tropics", data_dir="/nonexistent/path", seed=0)
    r2 = loader.load_region("tropics", data_dir="/nonexistent/path", seed=0)
    assert np.allclose(r1["T"], r2["T"])
    assert np.allclose(r1["q"], r2["q"])


def test_synthetic_fallback_differs_across_regions():
    loader = AIRSDataLoader()
    arctic = loader.load_region("arctic", data_dir="/nonexistent/path", seed=0)
    tropics = loader.load_region("tropics", data_dir="/nonexistent/path", seed=0)
    assert not np.allclose(arctic["T"], tropics["T"])


def test_synthetic_fallback_all_positive_for_loglog_compatibility():
    loader = AIRSDataLoader()
    for region in ("arctic", "tropics", "monsoon"):
        data = loader.load_region(region, data_dir="/nonexistent/path", seed=0)
        assert np.all(data["T"] > 0)
        assert np.all(data["q"] > 0)
