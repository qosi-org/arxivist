"""
Builds the labelled (signature, class) dataset for each of the paper's named
experiments (Sections 5.1-5.3, 6.1-6.3, 6.5-6.6, 6.8-6.9).
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from sig_vol_id.features.signatures import SignatureComputer
from sig_vol_id.simulators.heston import HestonSimulator
from sig_vol_id.simulators.ou import OUSimulator
from sig_vol_id.simulators.rbergomi import RoughBergomiSimulator

# Each experiment maps to: list of (class_name, generator_fn) pairs.
# generator_fn(n_paths, cfg, rng, T) -> [n_paths, n_steps+1] raw path array.

EXPERIMENTS = {
    # --- Section 5: fixed parameters ---
    "5.1": {"classes": ["Heston", "OU", "rB0.1", "rB0.3"], "random_params": False, "T": 0.1, "order": 4},
    "5.2": {"classes": ["Heston", "rB0.15", "rB0.4", "rB0.6"], "random_params": False, "T": 0.1, "order": 4},
    "5.3": {"classes": ["rB0.05", "rB0.15", "rB0.25", "rB0.35"], "random_params": False, "T": 0.1, "order": 4},
    # --- Section 6: random parameters ---
    "6.1": {"classes": ["Heston", "OU", "rB0.1", "rB0.3"], "random_params": True, "T": 0.1, "order": 4},
    "6.2": {"classes": ["Heston", "rB0.1", "rB0.2", "rB0.3"], "random_params": True, "T": 0.1, "order": 4},
    "6.3": {"classes": ["rB0.05", "rB0.15", "rB0.25", "rB0.35"], "random_params": True, "T": 0.1, "order": 4},
    "6.5_order3": {"classes": ["rB0.05", "rB0.15", "rB0.25", "rB0.35"], "random_params": True, "T": 0.1, "order": 3},
    "6.5_order5": {"classes": ["rB0.05", "rB0.15", "rB0.25", "rB0.35"], "random_params": True, "T": 0.1, "order": 5},
    "6.6_T0.2": {"classes": ["rB0.05", "rB0.15", "rB0.25", "rB0.35"], "random_params": True, "T": 0.2, "order": 4},
    "6.6_T0.4": {"classes": ["rB0.05", "rB0.15", "rB0.25", "rB0.35"], "random_params": True, "T": 0.4, "order": 4},
    "6.8_shared_dist": {"classes": ["Heston", "OU", "rB0.1", "rB0.3"], "random_params": True, "T": 0.1, "order": 4, "heston_ou_shared_dist": True},
    "6.8_low_nu": {"classes": ["Heston", "OU", "rB0.1", "rB0.3"], "random_params": True, "T": 0.1, "order": 4, "heston_ou_shared_dist": True, "nu_override": (0.01, 0.10)},
    "6.8_high_nu": {"classes": ["Heston", "OU", "rB0.1", "rB0.3"], "random_params": True, "T": 0.1, "order": 4, "heston_ou_shared_dist": True, "nu_fixed": 0.28},
}


def _rbergomi_H(class_name: str) -> float:
    return float(class_name.replace("rB", ""))


class ExperimentBuilder:
    """Builds train/test signature datasets for a named experiment.

    Args:
        cfg: Loaded Config object (see utils/config.py).
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, experiment_name: str, n_paths_per_class: int, n_test_per_class: int, seed: int):
        """Simulate paths, compute signatures, and split into train/test.

        Returns:
            (X_train, y_train, X_test, y_test, class_names)
        """
        if experiment_name not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment '{experiment_name}'. Options: {list(EXPERIMENTS)}")
        spec = EXPERIMENTS[experiment_name]
        rng = np.random.default_rng(seed)
        T = spec["T"]
        n_steps = self.cfg["simulation"]["n_steps"]

        heston_sim = HestonSimulator(n_steps=n_steps, T=T)
        ou_sim = OUSimulator(n_steps=n_steps, T=T)
        rb_sim = RoughBergomiSimulator(n_steps=n_steps, T=T)
        sig_computer = SignatureComputer(order=spec["order"])

        n_total = n_paths_per_class + n_test_per_class
        class_names = spec["classes"]
        raw_paths = {}

        rb_classes = [c for c in class_names if c.startswith("rB")]
        rb_H_list = [_rbergomi_H(c) for c in rb_classes]

        # Shared-noise rBergomi generation (Section 6.1 control)
        if rb_classes:
            rb_params = {"xi": self.cfg["rbergomi"]["fixed"]["xi"] if not spec["random_params"] else self.cfg["rbergomi"]["random"]["xi"]}
            if spec["random_params"]:
                rb_params["eta_range"] = self.cfg["rbergomi"]["random"]["eta_range"]
            else:
                rb_params["eta"] = self.cfg["rbergomi"]["fixed"]["eta"]
            shared = rb_sim.simulate_shared_noise(n_total, rb_H_list, rb_params, rng)
            for c, H in zip(rb_classes, rb_H_list):
                raw_paths[c] = shared[H]

        if "Heston" in class_names:
            if spec["random_params"]:
                if spec.get("heston_ou_shared_dist"):
                    # Section 6.8, footnote 7: "Both processes share identical
                    # distributions for the mean-reversion parameters kappa and
                    # theta. The diffusion coefficients are sampled from
                    # comparable ranges, the Heston volatility-of-volatility
                    # parameter nu being additionally constrained by the Feller
                    # condition." -- i.e. kappa/theta come from a shared range,
                    # but nu and sigma are each independently drawn from their
                    # OWN (comparable, but not identical) range. Forcing nu
                    # numerically equal to a sigma draw (an earlier bug here)
                    # breaks the paper's intended mechanism.
                    kappa = rng.uniform(*self.cfg["ou"]["random"]["kappa_range"], size=n_total)
                    theta = rng.uniform(*self.cfg["ou"]["random"]["theta_range"], size=n_total)
                    nu_low, nu_high = self.cfg["heston"]["random"]["nu_range"]
                    margin = self.cfg["heston"]["random"]["feller_safety_margin"]
                    nu_max = margin * np.sqrt(2 * kappa * theta)
                    nu_upper = np.minimum(nu_high, nu_max)
                    nu = rng.uniform(nu_low, np.maximum(nu_upper, nu_low + 1e-6))
                    h_params = {"X0": self.cfg["heston"]["fixed"]["X0"], "kappa": kappa, "theta": theta, "nu": nu}
                else:
                    h_params = HestonSimulator.sample_random_params(n_total, self.cfg["heston"], rng)
                # nu_override/nu_fixed apply ON TOP of whichever kappa/theta path was
                # taken above (paper Sec 6.8: "keeping all other parameters unchanged"
                # when varying nu -- i.e. the shared-distribution setup stays in place).
                if "nu_override" in spec:
                    h_params["nu"] = rng.uniform(*spec["nu_override"], size=n_total)
                elif "nu_fixed" in spec:
                    h_params["nu"] = np.full(n_total, spec["nu_fixed"])
            else:
                h_params = self.cfg["heston"]["fixed"]
            raw_paths["Heston"] = heston_sim.simulate(n_total, h_params, rng)

        if "OU" in class_names:
            if spec["random_params"]:
                if spec.get("heston_ou_shared_dist"):
                    kappa = rng.uniform(*self.cfg["ou"]["random"]["kappa_range"], size=n_total)
                    theta = rng.uniform(*self.cfg["ou"]["random"]["theta_range"], size=n_total)
                    sigma = rng.uniform(*self.cfg["ou"]["random"]["sigma_range"], size=n_total)
                    ou_params = {"X0": self.cfg["ou"]["fixed"]["X0"], "kappa": kappa, "theta": theta, "sigma": sigma}
                else:
                    ou_params = OUSimulator.sample_random_params(n_total, self.cfg["ou"], rng)
            else:
                ou_params = self.cfg["ou"]["fixed"]
            raw_paths["OU"] = ou_sim.simulate(n_total, ou_params, rng)

        # Compute signatures + assemble labelled dataset
        X_parts, y_parts = [], []
        for label_idx, cname in enumerate(class_names):
            feats = sig_computer.compute(raw_paths[cname], T=T)
            X_parts.append(feats)
            y_parts.append(np.full(feats.shape[0], label_idx))

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)

        # Stratified split matching n_test_per_class per class
        test_frac = n_test_per_class / n_total
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_frac, stratify=y, random_state=seed
        )
        return X_train, y_train, X_test, y_test, class_names
