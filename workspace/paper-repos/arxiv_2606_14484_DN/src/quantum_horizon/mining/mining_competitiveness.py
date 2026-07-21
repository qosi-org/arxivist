"""
Grover-vs-classical-proof-of-work mining competitiveness model.

Implements Section 2.2 and Figure 1 of arXiv:2606.14484: Bitcoin's June-2026
difficulty implies ~2^79 classical hashes per block; Grover needs ~2^40
iterations (~sqrt(2^79)), each a fault-tolerant reversible double-SHA-256
oracle of 474,168 T-gates. Effective hashrate scales linearly with gate
speed (calibrated to the Aggarwal et al. 2017 benchmark of 13.8 GH/s at
66.7 MHz), and K parallel quantum machines gain only a sqrt(K) speedup
(the "Grover parallelization wall"), versus linear K scaling for classical
ASIC farms.
"""

from __future__ import annotations

import numpy as np


class MiningCompetitivenessModel:
    """Effective quantum "hashrate" model for Bitcoin proof-of-work.

    Args:
        calibration_hashrate_ghs: benchmark hashrate in GH/s (paper: 13.8,
            from Aggarwal et al. 2017).
        calibration_gate_speed_mhz: benchmark gate speed in MHz (paper: 66.7).
    """

    def __init__(
        self, calibration_hashrate_ghs: float = 13.8, calibration_gate_speed_mhz: float = 66.7
    ) -> None:
        self.calibration_hashrate_ghs = calibration_hashrate_ghs
        self.calibration_gate_speed_mhz = calibration_gate_speed_mhz

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"MiningCompetitivenessModel(calibration="
            f"{self.calibration_hashrate_ghs}GH/s@{self.calibration_gate_speed_mhz}MHz)"
        )

    def effective_hashrate(self, gate_speed_ghz: np.ndarray) -> np.ndarray:
        """Effective single-machine hashrate (TH/s), linearly scaled from the
        calibration benchmark (Figure 1).

        H_eff(f) = H_benchmark * (f / f_benchmark)

        Args:
            gate_speed_ghz: gate speed(s) in GHz, scalar or array.

        Returns:
            Effective hashrate in TH/s, same shape as input.
        """
        gate_speed_mhz = np.asarray(gate_speed_ghz) * 1000.0
        hashrate_ghs = self.calibration_hashrate_ghs * (gate_speed_mhz / self.calibration_gate_speed_mhz)
        return hashrate_ghs / 1000.0  # GH/s -> TH/s

    def parallel_hashrate(self, gate_speed_ghz: float, n_machines: np.ndarray) -> np.ndarray:
        """Effective network hashrate for K quantum machines under the
        Grover parallelization wall: scales as sqrt(K), not K.

        Args:
            gate_speed_ghz: gate speed of each individual machine, in GHz.
            n_machines: K, scalar or array of machine counts.

        Returns:
            Effective combined hashrate in TH/s.
        """
        single = self.effective_hashrate(np.asarray(gate_speed_ghz))
        return single * np.sqrt(np.asarray(n_machines, dtype=float))

    def machines_for_target_hashrate(self, gate_speed_ghz: float, target_hashrate_ths: float) -> float:
        """Number of quantum machines K required to reach a target combined
        hashrate under sqrt(K) scaling.

        target = single * sqrt(K)  =>  K = (target / single)^2

        Args:
            gate_speed_ghz: gate speed of each machine, in GHz.
            target_hashrate_ths: desired combined hashrate in TH/s.

        Returns:
            Required number of machines K (float; ceil if an integer count
            is needed).
        """
        single = float(self.effective_hashrate(np.asarray(gate_speed_ghz)))
        return (target_hashrate_ths / single) ** 2

    def machines_for_51_percent(self, gate_speed_ghz: float, network_hashrate_ehs: float) -> float:
        """Number of quantum machines required to reach 51% of the given
        network hashrate (Section 2.2: "~7x10^13 quantum machines at 100 GHz").

        Args:
            gate_speed_ghz: gate speed of each machine, in GHz.
            network_hashrate_ehs: total network hashrate in EH/s.

        Returns:
            Required number of machines K.
        """
        target_ths = 0.51 * network_hashrate_ehs * 1e6  # EH/s -> TH/s
        return self.machines_for_target_hashrate(gate_speed_ghz, target_ths)
