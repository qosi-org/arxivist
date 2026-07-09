"""Feature normalization for market and private states.

Paper reference: Section 3.1 (state space), Appendix B.3 (feature
normalization) and Appendix A.2 (long-term average order book shapes used
to normalize volumes).
"""
from __future__ import annotations

import numpy as np


class FeatureNormalizer:
    """Builds the normalized flat state feature vector described in
    Appendix B.3.

    ASSUMED (confidence 0.7): the exact concatenation order of features is
    not prescribed by the paper; we use a fixed, documented order:
    [bid_price_norm, ask_price_norm, bid_vols_norm(K-1), ask_vols_norm(K-1),
     market_flow, limit_flow, cancel_flow, mid_price_drift,
     time_norm, inventory_norm, num_limit_orders_norm,
     levels_norm(M), queue_positions_norm(M), gamma(K-1)]
    """

    def __init__(self, K: int, M: int, avg_shape_bid: np.ndarray, avg_shape_ask: np.ndarray,
                 p_b0: float, p_a0: float):
        self.K = K
        self.M = M
        self.avg_shape_bid = avg_shape_bid  # long-term average volumes, level 1..K-1
        self.avg_shape_ask = avg_shape_ask
        self.p_b0 = p_b0
        self.p_a0 = p_a0

    def normalize_state(self, raw_state: dict) -> np.ndarray:
        """Convert a raw state dict (as produced by TradeExecutionEnv) into
        the normalized feature vector used as policy/value network input.
        """
        feats = []

        # Market features.
        feats.append((raw_state["best_bid"] - self.p_b0) / 10.0)
        feats.append((raw_state["best_ask"] - self.p_a0) / 10.0)

        bid_vols = np.asarray(raw_state["bid_volumes"][: self.K - 1], dtype=np.float64)
        ask_vols = np.asarray(raw_state["ask_volumes"][: self.K - 1], dtype=np.float64)
        eps = 1e-6
        feats.extend((bid_vols / (self.avg_shape_bid[: self.K - 1] + eps)).tolist())
        feats.extend((ask_vols / (self.avg_shape_ask[: self.K - 1] + eps)).tolist())

        dm = raw_state["market_flow"]
        dl = raw_state["limit_flow"]
        dc = raw_state["cancel_flow"]
        feats.append(np.clip(dm, -1.0, 1.0))
        feats.append(np.clip(dl, -1.0, 1.0))
        feats.append(np.clip(dc, -1.0, 1.0))

        prev_mid = raw_state.get("prev_mid_price", raw_state["mid_price"])
        drift = 0.0 if prev_mid == 0 else (raw_state["mid_price"] - prev_mid) / prev_mid
        feats.append(drift)

        # Private features.
        feats.append(raw_state["t"] / raw_state["T"])
        feats.append(raw_state["inventory"] / self.M)
        feats.append(raw_state["num_limit_orders"] / max(1, raw_state["inventory"]))

        levels = raw_state["order_levels"]  # length M, padded with K / -K sentinels
        queues = raw_state["order_queue_positions"]  # length M, padded with 50 / -50
        feats.extend((np.asarray(levels, dtype=np.float64) / self.K).tolist())
        feats.extend((np.asarray(queues, dtype=np.float64) / 50.0).tolist())

        gamma = raw_state["gamma"]  # length K, allocation per level of own orders
        feats.extend(np.asarray(gamma, dtype=np.float64).tolist())

        return np.asarray(feats, dtype=np.float32)

    def feature_dim(self) -> int:
        return 2 + 2 * (self.K - 1) + 4 + 3 + 2 * self.M + self.K
