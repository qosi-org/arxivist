"""Unit tests for quantum_horizon.survey (top-20 readiness survey)."""

from __future__ import annotations

import pytest

from quantum_horizon.survey import QuantumReadinessSurvey


CSV_PATH = "data/table3_readiness_ratings.csv"


def test_load_ratings_returns_expected_columns():
    qrs = QuantumReadinessSurvey()
    df = qrs.load_ratings(CSV_PATH)
    assert set(df.columns) == {"coin", "signature_scheme", "exposure_model", "pq_status", "rating"}


def test_load_ratings_missing_file_raises():
    qrs = QuantumReadinessSurvey()
    with pytest.raises(FileNotFoundError):
        qrs.load_ratings("/nonexistent/path.csv")


def test_no_coin_reaches_rating_5():
    """Paper: 'No top-20 coin reaches 5.'"""
    qrs = QuantumReadinessSurvey()
    df = qrs.load_ratings(CSV_PATH)
    stats = qrs.summary_stats(df)
    assert stats["no_coin_reaches_5"] is True
    assert stats["max_rating"] == pytest.approx(4.0)


def test_bitcoin_and_dogecoin_near_bottom():
    """Paper: 'Bitcoin and Dogecoin are among the furthest behind.'"""
    qrs = QuantumReadinessSurvey()
    df = qrs.load_ratings(CSV_PATH)
    btc_rating = df.loc[df["coin"] == "Bitcoin", "rating"].iloc[0]
    doge_rating = df.loc[df["coin"] == "Dogecoin", "rating"].iloc[0]
    median_rating = df["rating"].median()
    assert btc_rating <= median_rating
    assert doge_rating <= median_rating
    assert doge_rating == df["rating"].min()


def test_xrp_solana_zcash_are_leaders():
    """Paper: 'A few (XRP, Solana, Zcash) are noticeably ahead with working
    test versions.'"""
    qrs = QuantumReadinessSurvey()
    df = qrs.load_ratings(CSV_PATH)
    max_rating = df["rating"].max()
    for coin in ["XRP", "Solana", "Zcash"]:
        assert df.loc[df["coin"] == coin, "rating"].iloc[0] == pytest.approx(max_rating)


def test_summary_stats_rating_counts_sum_to_total():
    qrs = QuantumReadinessSurvey()
    df = qrs.load_ratings(CSV_PATH)
    stats = qrs.summary_stats(df)
    assert sum(stats["rating_counts"].values()) == len(df)
