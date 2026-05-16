"""
evaluation/cost_model.py — Transaction cost model.

Paper: AGENTICAITA (arxiv:2605.12532), Section 4.6, Eqs. 12-13
Applied retroactively to the DRY_RUN equity curve.

Round-trip cost decomposition (Eq. 13):
  C^rt_i = Q_i * P_i * f_taker              (exchange fee)
          + 0.5 * Q_i * |P_ask - P_bid|     (half-spread)  [NOTE: paper has 1/(2*Q_i) — see below]
          + lambda * sigma_i * sqrt(Q_i / (V_i * P_i))  (market impact)

WARNING: There is a potential typo in Eq. 13 of the paper. The half-spread term
is written as (1/(2*Q_i)) * |P_ask - P_bid|, which gives a per-trade cost that
decreases with size — unusual for a spread cost. The more standard formulation
would be (Q_i/2) * |P_ask - P_bid| (cost scales with size). We implement the
paper's literal formula; users should verify against paper intent.
[conf: 0.70] — ambiguity noted in SIR

lambda = 0.8: calibrated to crypto perpetuals liquidity profiles [conf: 0.85]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class CostScenario(str, Enum):
    ZERO = "zero"                  # DRY_RUN baseline
    CONSERVATIVE = "conservative"  # maker-only fees
    REALISTIC = "realistic"        # taker + spread
    ADVERSE = "adverse"            # illiquid long-tail


# Table 7 from the paper: cost scenarios applied to 139 trades, mean position $188
SCENARIO_ROUNDTRIP_PCT = {
    CostScenario.ZERO: 0.0000,
    CostScenario.CONSERVATIVE: 0.0004,  # 0.04%
    CostScenario.REALISTIC: 0.0010,     # 0.10%
    CostScenario.ADVERSE: 0.0020,       # 0.20%
}


@dataclass
class TradeInput:
    """Inputs required for per-trade cost calculation."""
    quantity: float         # Q_i: position size in base asset units
    price: float            # P_i: entry price
    f_taker: float          # exchange taker fee rate
    bid_price: float        # P_bid: best bid at order submission
    ask_price: float        # P_ask: best ask at order submission
    sigma: float            # sigma_i: realized 1-min volatility
    avg_volume: float       # V_i: avg traded volume over execution window
    lambda_impact: float = 0.8  # calibrated to crypto perps [conf: 0.85]


def compute_roundtrip_cost(trade: TradeInput) -> float:
    """
    Compute round-trip transaction cost C^rt_i for a single trade.

    Paper: Section 4.6, Eq. 13.

    C^rt_i = Q_i * P_i * f_taker
           + (1 / (2*Q_i)) * |P_ask - P_bid|     [paper's literal formula — see module docstring]
           + lambda * sigma_i * sqrt(Q_i / (V_i * P_i))

    Returns total round-trip cost in USD.
    """
    if trade.quantity <= 0 or trade.price <= 0:
        return 0.0

    # Exchange fee
    exchange_fee = trade.quantity * trade.price * trade.f_taker

    # Half-spread (paper's literal formula)
    if trade.quantity > 0:
        half_spread = (1.0 / (2.0 * trade.quantity)) * abs(trade.ask_price - trade.bid_price)
    else:
        half_spread = 0.0

    # Market impact (square-root model)
    vol_price_product = trade.avg_volume * trade.price
    if vol_price_product > 0:
        market_impact = trade.lambda_impact * trade.sigma * math.sqrt(trade.quantity / vol_price_product)
    else:
        market_impact = 0.0

    return exchange_fee + half_spread + market_impact


def sensitivity_analysis(
    net_pnl_dryrun: float,
    n_trades: int,
    mean_position_usd: float,
) -> dict:
    """
    Reproduce Table 7: sensitivity of net PnL to transaction costs.

    Paper: Section 6, Limitation L1, Table 7.
    Applied retroactively to dry-run PnL across three cost scenarios.

    Args:
        net_pnl_dryrun: Net PnL from DRY_RUN (zero-cost baseline).
        n_trades: Number of executed trades.
        mean_position_usd: Mean position size in USD per trade.

    Returns:
        Dict of scenario → {roundtrip_pct, total_cost, adj_net_pnl}.
    """
    results = {}
    for scenario, pct in SCENARIO_ROUNDTRIP_PCT.items():
        total_cost = pct * mean_position_usd * n_trades
        adj_pnl = net_pnl_dryrun - total_cost
        results[scenario.value] = {
            "roundtrip_pct": pct,
            "roundtrip_label": f"{pct*100:.2f}%",
            "total_cost_usd": -total_cost,
            "adj_net_pnl_usd": adj_pnl,
        }
    return results


def benchmark_comparison(
    system_pnl: float,
    btc_pnl: float,
    cash_pnl: float = 0.0,
) -> dict:
    """
    Reproduce Table 4: three-way benchmark comparison.

    Paper: Section 5, Table 4 and Section 6.

    Args:
        system_pnl: AGENTICAITA net PnL.
        btc_pnl: BTC buy-and-hold PnL (funding-adjusted).
        cash_pnl: Stablecoin/cash PnL (default 0).
    """
    return {
        "agenticaita": {"net_pnl": system_pnl},
        "btc_buyhold": {"net_pnl": btc_pnl},
        "cash": {"net_pnl": cash_pnl},
        "alpha_vs_btc_pp": round(
            (system_pnl - btc_pnl) / abs(btc_pnl) * 100 if btc_pnl != 0 else float("inf"), 4
        ),
        "alpha_vs_cash_pp": round(
            (system_pnl - cash_pnl) / 1.0 * 100 if cash_pnl == 0 else (system_pnl - cash_pnl) / abs(cash_pnl) * 100, 4
        ),
        "note": (
            "BTC benchmark is funding-adjusted; reported alpha is a conservative lower bound. "
            "Paper Section 5, Table 4 and Eq. (PnL^adj_bench = Delta_S_t * Q + sum(FundingRate_tk * Q * Delta_tk))."
        ),
    }
