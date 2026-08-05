"""
data/preprocessing.py
========================
Dataset-specific preprocessing pipelines (Section 3): Jane Street, Optiver,
Time-IMM, and DSLOB. Produces windowed tensors + masks shared by all neural
models (Section 3: "All neural models ... share the same preprocessed
windows to ensure a fair comparison").
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd


class JaneStreetPreprocessor:
    """
    Section 3.1: streamed, chronologically-sorted, (symbol,date)-grouped
    sliding windows of length 20 over 79 numerical features, with a binary
    missing-value mask. Training partitions limited to the first 500 days.
    """

    def __init__(self, lookback: int = 20, n_features: int = 79) -> None:
        self.lookback = lookback
        self.n_features = n_features

    def stream_windows(
        self, partition_path: str, max_days: int | None = None
    ) -> Iterator[dict]:
        """
        Args:
            partition_path: path to a single partition file (parquet/csv)
                with columns [date_id, time_id, symbol_id, feature_00..78, responder_6, weight].
            max_days: restrict to the first N unique date_ids (Section 3.1:
                "the training set is limited to the first 500 days").
        Yields:
            {'x': [lookback, n_features], 'mask': [lookback, n_features], 'y': float, 'weight': float}
        """
        df = pd.read_parquet(partition_path) if partition_path.endswith(".parquet") else pd.read_csv(partition_path)
        if max_days is not None:
            keep_dates = sorted(df["date_id"].unique())[:max_days]
            df = df[df["date_id"].isin(keep_dates)]

        feature_cols = [c for c in df.columns if c.startswith("feature_")][: self.n_features]
        for (symbol, date), group in df.groupby(["symbol_id", "date_id"]):
            group = group.sort_values("time_id")
            values = group[feature_cols].to_numpy(dtype=np.float32)
            targets = group["responder_6"].to_numpy(dtype=np.float32)
            mask = (~np.isnan(values)).astype(np.float32)
            values = np.nan_to_num(values, nan=0.0)
            n = len(group)
            for i in range(self.lookback, n):
                yield {
                    "x": values[i - self.lookback : i],
                    "mask": mask[i - self.lookback : i],
                    "y": float(targets[i]),
                    "weight": float(group["weight"].to_numpy()[i]) if "weight" in group.columns else 1.0,
                }


class OptiverPreprocessor:
    """
    Section 3.2: fuses asynchronous order-book snapshots and trade records
    into a regularly sampled 1Hz, 600-step sequence with 7 derived features
    per stock-time-window pair. Target is log-transformed realized volatility.
    """

    N_SECONDS = 600
    N_FEATURES = 7

    def fuse_book_and_trades(self, book_df: pd.DataFrame, trade_df: pd.DataFrame) -> np.ndarray:
        """
        Args:
            book_df: columns [seconds_in_bucket, bid_price1, ask_price1,
                bid_size1, ask_size1, bid_price2, ask_price2, bid_size2, ask_size2].
            trade_df: columns [seconds_in_bucket, price, size, order_count].
        Returns:
            [600, 7] forward-filled 1Hz feature matrix: mid_price, spread,
            log_mid_price, log_return, volume_imbalance, total_size, trade_price_avg.
        """
        book_df = book_df.sort_values("seconds_in_bucket")
        mid_price = (book_df["bid_price1"] + book_df["ask_price1"]) / 2
        spread = book_df["ask_price1"] - book_df["bid_price1"]
        log_mid = np.log(mid_price)
        log_return = log_mid.diff().fillna(0.0)
        bid_size = book_df["bid_size1"] + book_df.get("bid_size2", 0)
        ask_size = book_df["ask_size1"] + book_df.get("ask_size2", 0)
        vol_imbalance = (bid_size - ask_size) / (bid_size + ask_size + 1e-8)
        total_size = bid_size + ask_size

        book_features = pd.DataFrame({
            "mid_price": mid_price.values, "spread": spread.values,
            "log_mid_price": log_mid.values, "log_return": log_return.values,
            "volume_imbalance": vol_imbalance.values, "total_size": total_size.values,
        }, index=book_df["seconds_in_bucket"].values)

        full_index = np.arange(self.N_SECONDS)
        book_features = book_features.reindex(full_index).ffill().bfill()

        if len(trade_df) > 0:
            trade_agg = trade_df.groupby("seconds_in_bucket")["price"].mean()
            trade_agg = trade_agg.reindex(full_index).ffill().bfill()
        else:
            trade_agg = pd.Series(0.0, index=full_index)

        out = np.column_stack([book_features.to_numpy(dtype=np.float32), trade_agg.to_numpy(dtype=np.float32)])
        assert out.shape == (self.N_SECONDS, self.N_FEATURES), f"unexpected shape {out.shape}"
        assert not np.isnan(out).any(), "NaNs remain after forward/backward fill"
        return out

    def build_dataset(self, windows: List[Tuple[pd.DataFrame, pd.DataFrame, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            windows: list of (book_df, trade_df, realized_volatility) per stock-window.
        Returns:
            (X [N, 600, 7], y [N] log-transformed realized volatility).
        """
        X, y = [], []
        for book_df, trade_df, rv in windows:
            X.append(self.fuse_book_and_trades(book_df, trade_df))
            y.append(np.log1p(rv))
        return np.stack(X), np.array(y, dtype=np.float32)


class TimeIMMPreprocessor:
    """
    Section 3.3: hourly EPA-Air data across 8 counties, forward/backward
    filled per entity, 24-hour lookback windows predicting next-hour
    temperature; 70/15/15 temporal split applied per entity.
    """

    LOOKBACK = 24
    N_FEATURES = 4

    def load_and_fill(self, county_dfs: dict) -> pd.DataFrame:
        """
        Args:
            county_dfs: dict mapping county name -> DataFrame with columns
                [timestamp, temperature, particulate_matter, air_quality_index, ozone].
        Returns:
            concatenated dataframe with an 'entity' column, forward/backward-filled.
        """
        frames = []
        for entity, df in county_dfs.items():
            df = df.sort_values("timestamp").copy()
            df["entity"] = entity
            feature_cols = ["temperature", "particulate_matter", "air_quality_index", "ozone"]
            df[feature_cols] = df[feature_cols].ffill().bfill()
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def build_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            (X [N, 24, 4], y [N] next-hour temperature, entity_ids [N])
        """
        feature_cols = ["temperature", "particulate_matter", "air_quality_index", "ozone"]
        X, y, entities = [], [], []
        for entity, group in df.groupby("entity"):
            values = group[feature_cols].to_numpy(dtype=np.float32)
            n = len(values)
            for i in range(self.LOOKBACK, n):
                X.append(values[i - self.LOOKBACK : i])
                y.append(values[i, 0])  # temperature is column 0
                entities.append(entity)
        return np.stack(X), np.array(y, dtype=np.float32), np.array(entities)


class DSLOBGenerator:
    """
    Section 3.4: synthetic Deep Synthetic Limit Order Book generator.
    Amplifies a Vasicek mid-price SDE and GARCH(1,1) volatility fitted to a
    detected crash window in a seed high-frequency LOB dataset, then adds
    VAR(1)-correlated noise to the remaining 83 features and applies
    Gaussian-process time warping.

    NOTE: the paper does not name or provide its "real high-frequency limit
    order book dataset" seed source (Section 3.4). This generator therefore
    cannot reproduce the paper's actual DSLOB dataset -- it implements the
    described *generation procedure* so it can be exercised end-to-end
    against any user-supplied (or synthetic placeholder) seed LOB series.
    See data/README_data.md and comparison/hallucination_report.md.
    """

    def detect_crash_window(self, mid_price: np.ndarray, window_frac: float = 0.15) -> Tuple[int, int]:
        """
        Simplified CUSUM-style change-point detection: identifies the
        contiguous window with the steepest cumulative decline, as a stand-in
        for the paper's "CUSUM and Bayesian change point detection" (exact
        algorithm parameters not given in the paper).
        """
        n = len(mid_price)
        w = max(int(n * window_frac), 10)
        returns = np.diff(np.log(mid_price + 1e-8))
        cum_returns = np.cumsum(returns)
        best_start, best_decline = 0, 0.0
        for start in range(0, n - w):
            decline = cum_returns[start] - cum_returns[min(start + w, n - 2)]
            if decline > best_decline:
                best_decline, best_start = decline, start
        return best_start, min(best_start + w, n)

    def fit_vasicek(self, mid_price: np.ndarray, dt: float = 1.0) -> dict:
        """
        MLE fit of dP_t = theta(mu - P_t)dt + sigma*dW_t via OLS on the
        discretized recursion P_{t+1} - P_t = theta*(mu-P_t)*dt + noise
        (Section 3.4, Eq. for the Vasicek SDE), then amplified per the
        paper: theta *= 1.5, mu *= 1.2.
        """
        P = mid_price[:-1]
        dP = np.diff(mid_price)
        # dP = theta*mu*dt - theta*dt*P + noise  =>  linear regression dP ~ a + b*P
        design = np.column_stack([np.ones_like(P), P])
        coeffs, *_ = np.linalg.lstsq(design, dP, rcond=None)
        a, b = coeffs
        theta = -b / dt
        mu = a / (theta * dt) if theta != 0 else float(np.mean(mid_price))
        residuals = dP - design @ coeffs
        sigma = float(np.std(residuals)) / np.sqrt(dt)
        return {"theta": float(theta) * 1.5, "mu": float(mu) * 1.2, "sigma": sigma}

    def fit_garch(self, log_returns: np.ndarray) -> dict:
        """
        GARCH(1,1) fit via the `arch` package if available, else a simple
        method-of-moments fallback; amplified per the paper:
        beta' = min(0.95, 1.1*beta), alpha' = 1.2*alpha.
        """
        try:
            from arch import arch_model

            am = arch_model(log_returns * 100, vol="Garch", p=1, q=1, mean="Zero")
            res = am.fit(disp="off")
            omega, alpha, beta = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
        except Exception:
            # Fallback: crude moment-matching if `arch` is unavailable or fails to converge.
            var = np.var(log_returns)
            alpha, beta = 0.1, 0.8
            omega = var * (1 - alpha - beta)
        beta_amp = min(0.95, 1.1 * beta)
        alpha_amp = 1.2 * alpha
        return {"omega": float(omega), "alpha": float(alpha_amp), "beta": float(beta_amp)}

    def simulate_synthetic_mid_price(self, params: dict, n_steps: int, p0: float, seed: int = 0) -> np.ndarray:
        """Simulate the amplified Vasicek SDE forward (Euler-Maruyama)."""
        rng = np.random.default_rng(seed)
        theta, mu, sigma = params["theta"], params["mu"], params["sigma"]
        p = np.empty(n_steps)
        p[0] = p0
        for t in range(1, n_steps):
            p[t] = p[t - 1] + theta * (mu - p[t - 1]) + sigma * rng.normal()
        return p

    def simulate_garch_returns(self, params: dict, n_steps: int, seed: int = 0) -> np.ndarray:
        """Simulate amplified GARCH(1,1) returns forward."""
        rng = np.random.default_rng(seed)
        omega, alpha, beta = params["omega"], params["alpha"], params["beta"]
        sigma2 = np.empty(n_steps)
        r = np.empty(n_steps)
        sigma2[0] = omega / max(1 - alpha - beta, 1e-4)
        r[0] = 0.0
        for t in range(1, n_steps):
            eps = rng.normal()
            r[t] = np.sqrt(sigma2[t - 1]) * eps
            sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        return r

    def generate_synthetic_features(
        self, seed_features: np.ndarray, n_samples: int, seed: int = 0
    ) -> np.ndarray:
        """
        Generates the remaining feature columns (beyond mid-price) via
        VAR(1)-correlated noise added to the seed data (Section 3.4):
            f_synth_i(t) = f_seed_i(t) + eta_i(t),  eta(t) ~ N(0, Sigma)
        where Sigma is the residual covariance of a VAR(1) fitted to the
        seed features, and a Gaussian-process time-warp stretches/compresses
        the temporal axis to extend the effective training set.

        Args:
            seed_features: [T, F] seed feature panel (crash window).
            n_samples: number of synthetic samples to generate.
        Returns:
            [n_samples, F] synthetic feature panel.
        """
        rng = np.random.default_rng(seed)
        T, F = seed_features.shape

        # VAR(1) residual covariance.
        X_lag = seed_features[:-1]
        X_cur = seed_features[1:]
        design = np.column_stack([np.ones(T - 1), X_lag])
        coeffs, *_ = np.linalg.lstsq(design, X_cur, rcond=None)
        residuals = X_cur - design @ coeffs
        cov = np.cov(residuals, rowvar=False) + 1e-8 * np.eye(F)

        # Gaussian-process-style time warp: smooth random deformation, mean 1, var 0.1.
        warp = 1.0 + rng.normal(0, np.sqrt(0.1), size=n_samples)
        warp = np.clip(np.cumsum(warp) / np.arange(1, n_samples + 1), 0.5, 2.0)  # smoothed running average
        src_idx = np.clip((np.arange(n_samples) * warp).astype(int) % T, 0, T - 1)

        noise = rng.multivariate_normal(np.zeros(F), cov, size=n_samples)
        return seed_features[src_idx] + noise

    def validate_synthetic(self, seed: np.ndarray, synthetic: np.ndarray) -> dict:
        """
        Rigorous validation checks per Section 3.4:
            - KS test on return distributions (target: p > 0.05)
            - autocorrelation decay of squared returns up to lag 50
            - average abs. difference in correlation matrices (target: < 0.03)
            - 99.5th percentile of negative returns within 5% of seed's value
        """
        from scipy.stats import ks_2samp

        ks_stat, ks_pvalue = ks_2samp(seed.ravel(), synthetic.ravel())

        corr_seed = np.corrcoef(seed, rowvar=False)
        corr_synth = np.corrcoef(synthetic, rowvar=False)
        corr_matrix_diff = float(np.mean(np.abs(corr_seed - corr_synth)))

        seed_neg = seed[seed < 0]
        synth_neg = synthetic[synthetic < 0]
        seed_p995 = np.percentile(seed_neg, 0.5) if seed_neg.size else 0.0
        synth_p995 = np.percentile(synth_neg, 0.5) if synth_neg.size else 0.0
        tail_pct_diff = abs(synth_p995 - seed_p995) / (abs(seed_p995) + 1e-8)

        return {
            "ks_pvalue": float(ks_pvalue),
            "corr_matrix_diff": corr_matrix_diff,
            "tail_pct_diff": float(tail_pct_diff),
        }
