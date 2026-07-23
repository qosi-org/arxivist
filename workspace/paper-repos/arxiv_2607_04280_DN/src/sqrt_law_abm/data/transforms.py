"""
Metaorder reconstruction (Section 2.5) and per-day normalization (Section 3.1,
Eq. 2) of the impact analysis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sqrt_law_abm.models.lob import Trade


@dataclass
class ReconstructedMetaorder:
    """A metaorder reconstructed ex post from the trade tape (Section 2.5).

    Args:
        agent_id: Aggressor agent id.
        side: "buy" or "sell".
        total_size: Sum of executed sizes in this metaorder.
        t_start: Step of the first trade.
        t_end: Step of the last trade.
        day: Trading day index (t_start // steps_per_day).
        signed_impact: Signed price displacement from the pre-metaorder mid
            to the post-metaorder mid (sign convention: positive if impact is
            in the direction of the metaorder's side).
    """

    agent_id: int
    side: str
    total_size: float
    t_start: int
    t_end: int
    day: int
    signed_impact: float


class MetaorderReconstructor:
    """Groups consecutive same-agent, same-direction trades into metaorders,
    and normalizes them by per-day volume/range (Sections 2.5, 3.1).
    """

    def reconstruct(
        self,
        trade_tape: list[Trade],
        delta_t_ticks: int = 10,
        steps_per_day: int = 5000,
    ) -> list[ReconstructedMetaorder]:
        """Reconstruct metaorders from the trade tape.

        Two trades from the same agent are assigned to different metaorders
        if separated by more than `delta_t_ticks` steps or if the direction
        flips (Section 2.5).

        Args:
            trade_tape: Chronologically-ordered list of Trade.
            delta_t_ticks: Grouping threshold (Sec. 2.5: 10 steps).
            steps_per_day: Ticks per trading day, for day bucketing.

        Returns:
            List of ReconstructedMetaorder.
        """
        by_agent: dict[int, list[Trade]] = {}
        for tr in trade_tape:
            by_agent.setdefault(tr.aggressor_agent_id, []).append(tr)

        metaorders: list[ReconstructedMetaorder] = []
        for agent_id, trades in by_agent.items():
            trades = sorted(trades, key=lambda tr: tr.t)
            current_group: list[Trade] = []

            def flush(group: list[Trade]) -> None:
                if not group:
                    return
                total_size = sum(tr.size for tr in group)
                t_start, t_end = group[0].t, group[-1].t
                # Impact = displacement of last trade price from the price
                # just before the group started (proxy: first trade's price).
                pre_price = group[0].price
                post_price = group[-1].price
                side = group[0].aggressor_side
                sign = 1.0 if side == "buy" else -1.0
                signed_impact = sign * (post_price - pre_price)
                metaorders.append(
                    ReconstructedMetaorder(
                        agent_id=agent_id,
                        side=side,
                        total_size=total_size,
                        t_start=t_start,
                        t_end=t_end,
                        day=t_start // steps_per_day,
                        signed_impact=signed_impact,
                    )
                )

            for tr in trades:
                if not current_group:
                    current_group = [tr]
                    continue
                last = current_group[-1]
                gap = tr.t - last.t
                same_direction = tr.aggressor_side == last.aggressor_side
                if gap <= delta_t_ticks and same_direction:
                    current_group.append(tr)
                else:
                    flush(current_group)
                    current_group = [tr]
            flush(current_group)

        return metaorders

    def normalize(
        self,
        metaorders: list[ReconstructedMetaorder],
        daily_volume: dict[int, float],
        daily_range: dict[int, float],
        min_qnorm_threshold_pct: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize (Q, I) pairs by per-day volume/range (Eq. 2).

        Metaorders with Q/V_D below `min_qnorm_threshold_pct` are dropped
        (Section 3.1: "dominated by the bid-ask spread rather than by the
        shape of the LOB").

        Args:
            metaorders: Reconstructed metaorders.
            daily_volume: day -> V_D.
            daily_range: day -> sigma_D.
            min_qnorm_threshold_pct: Drop threshold in percent (Sec 3.1: 0.01%).

        Returns:
            (q_norm, i_norm): 1D arrays of normalized size and impact,
            filtered and with non-positive impacts removed (log-log fit
            requires positive values).
        """
        q_norm_list = []
        i_norm_list = []
        for mo in metaorders:
            v_d = daily_volume.get(mo.day)
            sigma_d = daily_range.get(mo.day)
            if not v_d or not sigma_d:
                continue
            q_norm = mo.total_size / v_d
            if q_norm * 100 < min_qnorm_threshold_pct:
                continue
            i_norm = mo.signed_impact / sigma_d
            if i_norm <= 0:
                continue
            q_norm_list.append(q_norm)
            i_norm_list.append(i_norm)

        return np.array(q_norm_list), np.array(i_norm_list)
