"""Unit tests for quantum_horizon.exposure (Bitcoin and Ethereum exposure models)."""

from __future__ import annotations

import pytest

from quantum_horizon.exposure import BitcoinExposureModel, EthereumExposureModel


def test_bitcoin_reconcile_sources_simple_mean():
    btc = BitcoinExposureModel()
    result = btc.reconcile_sources({"a": 0.2, "b": 0.3, "c": 0.4})
    assert result == pytest.approx(0.3)


def test_bitcoin_reconcile_sources_unsupported_method_raises():
    btc = BitcoinExposureModel()
    with pytest.raises(ValueError):
        btc.reconcile_sources({"a": 0.2}, method="weighted")


def test_bitcoin_decompose_matches_paper_figures():
    """Paper Figure 3: irreducible ~2.3M (12%), migratable ~3.7M (19%),
    protected ~13.4M (68%)."""
    btc = BitcoinExposureModel()
    result = btc.decompose(19.4, 2.3, 3.7)
    assert result["irreducible_btc"] == pytest.approx(2.3)
    assert result["migratable_btc"] == pytest.approx(3.7)
    assert result["protected_btc"] == pytest.approx(13.4, abs=0.01)
    assert result["irreducible_frac"] == pytest.approx(0.1186, abs=0.01)
    assert result["migratable_frac"] == pytest.approx(0.1907, abs=0.01)
    assert result["protected_frac"] == pytest.approx(0.6907, abs=0.01)


def test_bitcoin_decompose_fractions_sum_to_one():
    btc = BitcoinExposureModel()
    result = btc.decompose(19.4, 2.3, 3.7)
    total_frac = result["irreducible_frac"] + result["migratable_frac"] + result["protected_frac"]
    assert total_frac == pytest.approx(1.0)


def test_ethereum_top_down_estimate():
    eth = EthereumExposureModel()
    result = eth.top_down_estimate(staked_fraction=0.32, contract_fraction=0.08)
    assert result == pytest.approx(0.60)


def test_ethereum_top_down_within_paper_range():
    """Paper: top-down composition gives ~55-63%."""
    eth = EthereumExposureModel()
    result = eth.top_down_estimate(staked_fraction=0.32, contract_fraction=0.08)
    assert 0.55 <= result <= 0.63


def test_ethereum_bottom_up_within_paper_range():
    """Paper: bottom-up on-chain build gives ~45-55% after correction."""
    eth = EthereumExposureModel()
    result = eth.bottom_up_estimate(naive_beacon_overcount_fraction=0.21, correction_factor=2.3)
    assert 0.45 <= result <= 0.55


def test_ethereum_reconcile_returns_ordered_tuple():
    eth = EthereumExposureModel()
    low, high = eth.reconcile(0.6, 0.48)
    assert low < high
    assert low == pytest.approx(0.48)
    assert high == pytest.approx(0.6)


def test_ethereum_reconciled_range_within_paper_consensus():
    """Paper: consensus range 50-65%, most defensibly 55-60%."""
    eth = EthereumExposureModel()
    top_down = eth.top_down_estimate(0.32, 0.08)
    bottom_up = eth.bottom_up_estimate(0.21, 2.3)
    low, high = eth.reconcile(top_down, bottom_up)
    assert 0.45 <= low <= 0.65
    assert 0.45 <= high <= 0.65
