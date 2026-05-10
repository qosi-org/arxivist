"""
Dynamic Ising Model for network adoption inference.
Implements Stage 1 of the Q-Ising pipeline.

This module implements the node-wise conditional logistic model described in
Section 3.1 of arXiv:2605.06564, including:
  - Linear predictor (Eq. eq_linear_predictor)
  - Spike-and-slab prior (Eq. eq_spike_slab)
  - EMVS point estimation (Section 3.1)
  - MCMC posterior sampling via PyMC NUTS (Section 3.3)

Paper: "Dynamic Treatment on Networks" (arXiv:2605.06564), Section 3.1.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import expit  # sigmoid

from q_ising.data.network import NetworkData
from q_ising.data.panel import ObservationalPanel
from q_ising.utils.config import IsingConfig


class NodeIsingParams:
    """Parameters for a single node's Ising model.

    Stores beta coefficients and coupling parameters (gamma) as defined
    in Section 3.1, Eq. eq_linear_predictor.

    Args:
        node: Node index.
        bin_k: Bin index for this node.
        n_neighbor_bins: Number of distinct bins among neighbors.
        neighbor_bin_ids: Bin index for each neighbor (ordered by neighbor list).
    """

    def __init__(
        self,
        node: int,
        bin_k: int,
        neighbor_bin_ids: List[int],
    ) -> None:
        self.node = node
        self.bin_k = bin_k
        self.neighbor_bin_ids = neighbor_bin_ids

        # beta_0_k: intercept for bin k (Section 3.1)
        self.beta_0: float = 0.0
        # beta_1_k: direct treatment effect for bin k
        self.beta_1: float = 0.0
        # beta_2_k: persistence of past adoption
        self.beta_2: float = 0.0
        # beta_3_k: spillover from treating a neighbor
        self.beta_3: float = 0.0
        # gamma_{k, m_j}: coupling from neighbor j's bin m_j to node i's bin k
        # Indexed by position in neighbor list
        self.gamma: np.ndarray = np.zeros(len(neighbor_bin_ids))

    def to_vector(self) -> np.ndarray:
        """Return all parameters as a flat array [beta_0, beta_1, beta_2, beta_3, *gamma]."""
        return np.concatenate([[self.beta_0, self.beta_1, self.beta_2, self.beta_3], self.gamma])

    def from_vector(self, v: np.ndarray) -> None:
        """Load parameters from flat array."""
        self.beta_0, self.beta_1, self.beta_2, self.beta_3 = v[:4]
        self.gamma = v[4:]


class DynamicIsingModel:
    """Bayesian Dynamic Ising Model for network adoption inference.

    Implements the conditional logistic model from Section 3.1.
    Supports two estimation backends:
      - EMVS: fast MAP via EM-based variable selection (Rockova & George 2014)
      - MCMC: full posterior via PyMC NUTS (Hoffman & Gelman 2014)

    The linear predictor for node i in bin B_k at time t is (Eq. eq_linear_predictor):
        eta_{i,t}(a_t; theta_i) = beta_0_k
                                  + beta_1_k * 1[a_t == i]
                                  + beta_2_k * y_{i,t-1}
                                  + beta_3_k * 1[a_t in N_i]
                                  + sum_{j in N_i} gamma_{k, m_j} * y_{j,t-1}

    Args:
        network: NetworkData with bin assignments.
        config: IsingConfig with prior hyperparameters.
    """

    def __init__(self, network: NetworkData, config: IsingConfig) -> None:
        assert network.bin_labels is not None, "Bin labels must be assigned before fitting Ising model"
        self.network = network
        self.config = config
        self.K = network.K
        self.N = network.N

        # Fitted parameters: one NodeIsingParams per node
        self._params: Optional[List[NodeIsingParams]] = None
        # MCMC draws: list of P dicts mapping node -> parameter vector
        self._mcmc_draws: Optional[List[Dict[int, np.ndarray]]] = None
        self._fitted = False

    def _make_node_params(self, node: int) -> NodeIsingParams:
        neighbors = self.network.get_neighbors(node)
        neighbor_bins = [self.network.get_bin(j) for j in neighbors]
        return NodeIsingParams(node, self.network.get_bin(node), neighbor_bins)

    def linear_predictor(
        self,
        y_prev: np.ndarray,
        action: Optional[int],
        node: int,
        params: NodeIsingParams,
    ) -> float:
        """Compute the linear predictor eta_{i,t} for node i.

        Implements Eq. eq_linear_predictor from Section 3.1.
        Setting action=None corresponds to the counterfactual no-intervention
        state (a_t = empty) used for state construction (Eq. eq_latent_state).

        Args:
            y_prev: Previous adoption state [N].
            action: Treated node index, or None for counterfactual.
            node: Target node i.
            params: NodeIsingParams for node i.

        Returns:
            Scalar linear predictor value.
        """
        # Intercept (beta_0_k)
        eta = params.beta_0

        # Direct treatment effect: beta_1_k * 1[a_t == i] (Eq. eq_linear_predictor)
        if action is not None and action == node:
            eta += params.beta_1

        # Persistence: beta_2_k * y_{i,t-1} (Eq. eq_linear_predictor)
        eta += params.beta_2 * float(y_prev[node])

        # Neighbor spillover: beta_3_k * 1[a_t in N_i] (Eq. eq_linear_predictor)
        neighbors = self.network.get_neighbors(node)
        if action is not None and action in neighbors:
            eta += params.beta_3

        # Peer influence: sum_{j in N_i} gamma_{k, m_j} * y_{j,t-1} (Eq. eq_linear_predictor)
        for idx, j in enumerate(neighbors):
            eta += params.gamma[idx] * float(y_prev[j])

        return eta

    def adoption_prob(
        self,
        y_prev: np.ndarray,
        action: Optional[int],
        node: int,
        params: Optional[NodeIsingParams] = None,
    ) -> float:
        """Compute P(y_{i,t}=1 | ...) = sigmoid(eta_{i,t}).

        Implements Eq. eq_adoption_prob.

        Args:
            y_prev: Previous state [N].
            action: Treated node or None for counterfactual.
            node: Target node i.
            params: NodeIsingParams; uses fitted params if None.

        Returns:
            Adoption probability in (0, 1).
        """
        if params is None:
            assert self._fitted, "Model must be fitted before calling adoption_prob"
            params = self._params[node]
        eta = self.linear_predictor(y_prev, action, node, params)
        return float(expit(eta))

    def counterfactual_prob(
        self,
        y_prev: np.ndarray,
        node: int,
        param_dict: Optional[Dict] = None,
    ) -> float:
        """Compute l_hat^0_{i,t} — no-intervention adoption probability.

        Implements Eq. eq_latent_state: l_hat^0_{i,t} = sigma(eta_{i,t}(empty; theta_hat)).
        Setting action=None zeroes out all treatment indicators (Section 3.1).

        Args:
            y_prev: Previous state [N].
            node: Target node i.
            param_dict: Optional parameter override (for ensemble draws).

        Returns:
            Counterfactual probability in (0, 1).
        """
        if param_dict is not None:
            p = NodeIsingParams(node, self.network.get_bin(node),
                                [self.network.get_bin(j) for j in self.network.get_neighbors(node)])
            p.from_vector(param_dict[node])
        else:
            assert self._fitted
            p = self._params[node]
        return self.adoption_prob(y_prev, action=None, node=node, params=p)

    def _build_design_matrix(
        self, panel: ObservationalPanel
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Build per-node design matrices X_i and label vectors Y_i from panel.

        Returns:
            features: Dict[node -> np.ndarray [T, 4+|N_i|]]
            labels: Dict[node -> np.ndarray [T]]
        """
        N, T = self.N, panel.T
        features: Dict[int, List] = {i: [] for i in range(N)}
        labels: Dict[int, List] = {i: [] for i in range(N)}

        for t in range(1, T + 1):
            y_prev, a_t, y_t = panel.get_period(t)
            for node in range(N):
                neighbors = self.network.get_neighbors(node)
                # Features: [1, 1[a==i], y_{i,t-1}, 1[a in N_i], y_{j,t-1} for j in N_i]
                row = [
                    1.0,                                   # intercept
                    float(a_t == node),                    # direct treatment
                    float(y_prev[node]),                   # persistence
                    float(a_t in neighbors),               # neighbor spillover
                ]
                row += [float(y_prev[j]) for j in neighbors]  # peer influence
                features[node].append(row)
                labels[node].append(int(y_t[node]))

        feat_arrays = {i: np.array(features[i]) for i in range(N)}
        label_arrays = {i: np.array(labels[i]) for i in range(N)}
        return feat_arrays, label_arrays

    def fit_emvs(self, panel: ObservationalPanel) -> None:
        """Fit the Ising model via EMVS (MAP estimation).

        EMVS (Rockova & George 2014) alternates between:
          E-step: compute posterior inclusion probabilities z_{ij}
          M-step: solve weighted penalized logistic regression

        # WARNING: low-confidence implementation (confidence: 0.62)
        # TODO: Verify exact EM update equations against Rockova & George (2014).
        # Current implementation uses L1/L2 mixture via sklearn as a proxy.

        Args:
            panel: ObservationalPanel with training data.
        """
        from sklearn.linear_model import LogisticRegression

        features, labels = self._build_design_matrix(panel)
        self._params = []

        for node in range(self.N):
            p = self._make_node_params(node)
            X = features[node]   # [T, 4 + |N_i|]
            y = labels[node]     # [T]

            n_features = X.shape[1]
            n_coupling = len(self.network.get_neighbors(node))

            # EMVS approximation: fit L1-penalized logistic regression
            # The spike-slab prior encourages sparsity in coupling params (gamma)
            # ASSUMED: using L1 penalty as EMVS proxy (confidence: 0.62)
            if len(np.unique(y)) < 2:
                # All-zero or all-one labels: set params to zero (uninformative)
                p.beta_0 = 0.0
                p.beta_1 = 0.0
                p.beta_2 = 0.0
                p.beta_3 = 0.0
                p.gamma = np.zeros(n_coupling)
            else:
                try:
                    clf = LogisticRegression(
                        penalty="l1",
                        C=1.0 / (1.0 / self.config.v1),   # C = 1/lambda ~ slab variance
                        solver="liblinear",
                        fit_intercept=False,  # intercept is first feature column
                        max_iter=self.config.emvs_n_iter * 100,
                        random_state=0,
                    )
                    clf.fit(X, y)
                    coef = clf.coef_[0]
                    p.beta_0 = float(coef[0])
                    p.beta_1 = float(coef[1])
                    p.beta_2 = float(coef[2])
                    p.beta_3 = float(coef[3])
                    p.gamma = coef[4:].astype(float)
                except Exception:
                    # Fallback to zero initialization (uninformative baseline)
                    p.beta_0 = p.beta_1 = p.beta_2 = p.beta_3 = 0.0
                    p.gamma = np.zeros(n_coupling)

            self._params.append(p)

        self._fitted = True

    def fit_mcmc(self, panel: ObservationalPanel, seed: int = 42) -> None:
        """Fit the Ising model via MCMC (HMC/NUTS) for posterior sampling.

        Uses PyMC with NUTS sampler (Hoffman & Gelman 2014).
        Produces self.config.mcmc_n_draws posterior draws per node.

        # WARNING: low-confidence implementation (confidence: 0.65)
        # TODO: Verify HMC library matches paper (PyMC assumed; could be NumPyro/Stan).

        Args:
            panel: ObservationalPanel with training data.
            seed: Random seed for MCMC.
        """
        try:
            import pymc as pm
        except ImportError:
            raise ImportError("PyMC is required for MCMC estimation. pip install pymc")

        features, labels = self._build_design_matrix(panel)
        self._mcmc_draws = []
        all_draws: List[Dict[int, np.ndarray]] = [{} for _ in range(self.config.mcmc_n_draws)]

        for node in range(self.N):
            X = features[node]
            y = labels[node]
            n_beta = 4
            n_coupling = X.shape[1] - n_beta

            with pm.Model():
                # Priors for beta parameters: N(0, tau^2) — Section 3.1
                beta = pm.Normal("beta", mu=0, sigma=np.sqrt(self.config.tau_sq), shape=n_beta)

                # Spike-and-slab prior for coupling (gamma) — Eq. eq_spike_slab
                # ASSUMED: using continuous relaxation (BernoulliMixture) as MCMC proxy
                if n_coupling > 0:
                    K_bin = self.network.K
                    incl_prob = min(self.config.c / max(K_bin, 1), 0.99)
                    z = pm.Bernoulli("z", p=incl_prob, shape=n_coupling)
                    sigma_mix = z * np.sqrt(self.config.v1) + (1 - z) * np.sqrt(self.config.v0)
                    gamma = pm.Normal("gamma", mu=0, sigma=sigma_mix, shape=n_coupling)
                    theta = pm.math.concatenate([beta, gamma])
                else:
                    theta = beta

                # Likelihood (Eq. eq_likelihood)
                eta = pm.math.dot(X, theta)
                pm.Bernoulli("y_obs", logit_p=eta, observed=y)

                trace = pm.sample(
                    draws=self.config.mcmc_n_draws,
                    tune=self.config.mcmc_n_tune,
                    target_accept=0.9,
                    random_seed=seed + node,
                    progressbar=False,
                )

            beta_samples = trace.posterior["beta"].values.reshape(-1, n_beta)
            if n_coupling > 0:
                gamma_samples = trace.posterior["gamma"].values.reshape(-1, n_coupling)
            else:
                gamma_samples = np.zeros((len(beta_samples), 0))

            n_draws = min(self.config.mcmc_n_draws, len(beta_samples))
            for d in range(n_draws):
                param_vec = np.concatenate([beta_samples[d], gamma_samples[d]])
                all_draws[d][node] = param_vec

        self._mcmc_draws = all_draws
        # Also set point estimate (posterior mean) as _params for non-ensemble use
        self._params = []
        for node in range(self.N):
            p = self._make_node_params(node)
            mean_vec = np.mean([all_draws[d][node] for d in range(len(all_draws))], axis=0)
            p.from_vector(mean_vec)
            self._params.append(p)
        self._fitted = True

    def get_mcmc_draws(self) -> List[Dict[int, np.ndarray]]:
        """Return list of MCMC parameter draws.

        Returns:
            List of P dicts: [{node -> param_vector}, ...].
        """
        assert self._mcmc_draws is not None, "MCMC not yet run. Call fit_mcmc() first."
        return self._mcmc_draws

    def __repr__(self) -> str:
        return (
            f"DynamicIsingModel(N={self.N}, K={self.K}, "
            f"fitted={self._fitted}, method={self.config.estimation_method})"
        )
