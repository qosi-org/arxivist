"""
src/geomherd/pipeline/geomherd_pipeline.py
Top-level GeomHerd pipeline: graph -> ORC -> Ricci flow -> V_eff -> detectors.
Paper: arXiv:2605.11645, Section 2 (full pipeline)

Produces the GeomHerd output quadruplet:
    (kappa_bar_plus_OR, beta_minus, tau_sing, V_eff)
and fires alarms A_plus and A_minus.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from geomherd.detection.cusum import ContagionDetector, HerdingDetector
from geomherd.geometry.ricci_curvature import OllivierRicciComputer
from geomherd.geometry.ricci_flow import DiscreteRicciFlow
from geomherd.geometry.vocabulary import FSQVocabularyTracker
from geomherd.graph.agent_graph import AgentGraph
from geomherd.utils.config import GeomHerdConfig


@dataclass
class GeomHerdOutput:
    """Output at each snapshot step."""
    t: int
    kappa_bar_plus: float      # mean ORC over E+ (herding signal)
    beta_minus: float          # fraction of E- edges (contagion signal)
    tau_sing: float            # Ricci-flow neckpinch time (forward-looking)
    V_eff: float               # effective vocabulary (behavioral diversity)
    kappa_bar_all: float       # mean ORC over all edges
    n_edges: int               # number of retained edges
    alarm_plus: bool           # herding alarm (kappa_bar_plus CUSUM)
    alarm_minus: bool          # contagion alarm (beta_minus CUSUM+Kendall)
    cusum_S_plus: float        # CUSUM statistic for herding
    cusum_S_minus: float       # CUSUM statistic for contagion


class GeomHerdPipeline:
    """
    Full GeomHerd forward-looking herding detection pipeline.

    Paper reference: Section 2 (complete pipeline)

    Pipeline stages (per snapshot):
      1. AgentGraph: windowed action-agreement graph G_t (Eq. 1)
      2. OllivierRicciComputer: kappa_OR per edge (Eqs. 2-3), sign decomp
      3. DiscreteRicciFlow: tau_sing from fresh flow on G_t (Section 2.4)
      4. FSQVocabularyTracker: V_eff = exp(H(p_t)) (Section 2.4)
      5. HerdingDetector: CUSUM alarm on kappa_bar_plus (Eq. 4)
      6. ContagionDetector: CUSUM+Kendall alarm on beta_minus (Eqs. 5-6)

    Args:
        cfg: GeomHerdConfig with all hyperparameters
    """

    def __init__(self, cfg: Optional[GeomHerdConfig] = None):
        self.cfg = cfg or GeomHerdConfig()
        gc = self.cfg.graph
        cc = self.cfg.curvature
        rc = self.cfg.ricci_flow
        dc = self.cfg.detection
        vc = self.cfg.vocabulary

        self.agent_graph = AgentGraph(
            N=self.cfg.simulation.cws.N_agents,
            Tw=gc.Tw,
            w0=gc.w0,
            delta_t=gc.delta_t,
        )
        self.orc = OllivierRicciComputer(
            alpha=cc.alpha,
            kappa_plus_thresh=cc.kappa_plus_thresh,
            kappa_minus_thresh=cc.kappa_minus_thresh,
        )
        self.ricci_flow = DiscreteRicciFlow(
            orc_computer=self.orc,
            step_size=rc.step_size,
            max_iter=rc.max_iter,
            neckpinch_threshold=rc.neckpinch_threshold,
            flow_variant=rc.flow_variant,
        )
        self.vocab_tracker = FSQVocabularyTracker(
            codebook_dims=vc.codebook_dims,
            levels_per_dim=vc.levels_per_dim,
        )
        self.herding_detector = HerdingDetector(
            operating_point=dc.operating_point,
            baseline_window=dc.baseline_window,
            skip_initial=dc.skip_initial,
        )
        self.contagion_detector = ContagionDetector()
        self._history: List[GeomHerdOutput] = []
        self._snapshot_count: int = 0

    def step(self, actions: np.ndarray, t: int) -> Optional[GeomHerdOutput]:
        """
        Process one simulator step.

        Args:
            actions: [N] int array of agent actions for this step
            t: Current simulator step index
        Returns:
            GeomHerdOutput if a new snapshot was computed, else None
        """
        snapshot_ready = self.agent_graph.push(actions)
        if not snapshot_ready:
            return None

        self._snapshot_count += 1
        W = self.agent_graph.get_weight_matrix()
        G_sparse = self.agent_graph.get_sparse_graph()
        edge_list = self.agent_graph.get_edge_list()

        # Stage 2: ORC computation
        if not edge_list:
            kappa_dict: Dict = {}
        else:
            kappa_dict = self.orc.compute(W, edge_list)

        kappa_bar_plus = self.orc.mean_curvature_plus(kappa_dict)
        kappa_bar_all = self.orc.mean_curvature_all(kappa_dict)
        beta_minus = self.orc.beta_minus(kappa_dict)

        # Stage 3: Ricci flow singularity time
        tau_sing = self.ricci_flow.singularity_time(W)

        # Stage 4: Effective vocabulary
        V_eff = self.vocab_tracker.effective_vocab(actions)

        # Stage 5: Herding detector (kappa_bar_plus CUSUM)
        S_plus, alarm_plus = self.herding_detector.update(kappa_bar_plus)

        # Stage 6: Contagion detector (beta_minus CUSUM + Kendall-tau)
        S_minus, _, alarm_minus = self.contagion_detector.update(beta_minus)

        output = GeomHerdOutput(
            t=t,
            kappa_bar_plus=kappa_bar_plus,
            beta_minus=beta_minus,
            tau_sing=tau_sing,
            V_eff=V_eff,
            kappa_bar_all=kappa_bar_all,
            n_edges=len(edge_list),
            alarm_plus=alarm_plus,
            alarm_minus=alarm_minus,
            cusum_S_plus=S_plus,
            cusum_S_minus=S_minus,
        )
        self._history.append(output)
        return output

    def get_triplet(self, t: int) -> Tuple[float, float, float]:
        """
        Return the GeomHerd triplet (kappa_bar_OR, tau_sing, V_eff) for forecasting.
        Paper: Section 3.3.3 — conditioning feature for Kronos head.
        """
        if not self._history:
            return 0.0, float(self.ricci_flow.max_iter), self.vocab_tracker.K
        last = self._history[-1]
        return last.kappa_bar_all, last.tau_sing, last.V_eff

    def reset(self) -> None:
        """Reset all state (call between trajectories)."""
        self.agent_graph.reset()
        self.herding_detector.reset()
        self.contagion_detector.reset()
        self._history.clear()
        self._snapshot_count = 0

    @property
    def history(self) -> List[GeomHerdOutput]:
        return self._history

    def first_alarm_time(self, alarm_type: str = "plus") -> Optional[int]:
        """Return simulator step of first alarm, or None if no alarm fired."""
        attr = f"alarm_{alarm_type}"
        for out in self._history:
            if getattr(out, attr, False):
                return out.t
        return None

    def __repr__(self) -> str:
        return (f"GeomHerdPipeline(N_agents={self.agent_graph.N}, "
                f"snapshots={self._snapshot_count}, "
                f"op={self.herding_detector.operating_point})")
