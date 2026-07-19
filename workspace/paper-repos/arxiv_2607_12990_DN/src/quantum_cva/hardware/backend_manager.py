"""
IBM Quantum hardware execution wrapper.

Implements Appendix A.2 of arXiv:2607.12990: fixed-layout transpilation
targeting a heavy-hex backend with zero SWAP insertions, and an optional
Q-CTRL Performance Management Qiskit Function execution path for
error-suppressed hardware runs.

IMPORTANT (see architecture_plan.json risk_assessment, "High" severity item):
Q-CTRL's Performance Management workflow is a proprietary, managed service.
Its internal transpilation/error-suppression steps are not disclosed by the
paper, so `run_with_qctrl` here is a thin, clearly-labelled wrapper around
the real Q-CTRL Qiskit Function *when credentials are configured*, with an
explicit, documented fallback that replays the paper's own published
calibration data (Tables 17-18) when no such credentials are available. This
fallback does NOT claim to reproduce the exact hardware execution -- it
reproduces the *statistics* the paper reports, via
`estimation.contrast_calibration.ContrastCalibrator.build_hardware_replay_model`.

SIR reference: architecture.modules "A_Theta" (execution target),
implementation_assumptions[3] (Q-CTRL black-box), risk_assessment[0].
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from qiskit import QuantumCircuit, transpile
from qiskit.providers import Backend
from qiskit_aer import AerSimulator


class HardwareUnavailableError(RuntimeError):
    """Raised when hardware or Q-CTRL execution is requested but no
    credentials/connectivity are configured, and no fallback was permitted."""


class BackendManager:
    """Manages access to IBM Quantum backends, local Aer simulators, and the
    Q-CTRL Performance Management execution path.

    Args:
        backend_name: name of the target IBM Quantum backend (paper uses
            "ibm_basquecountry", a 156-qubit Heron r2 processor accessed via
            Basque Quantum/BasQ, Appendix A.2).
        simulator_fallback: which local simulator to use when no IBM Quantum
            credentials are configured ("aer_statevector" or "aer_noisy").
    """

    def __init__(
        self, backend_name: str = "ibm_basquecountry", simulator_fallback: str = "aer_statevector"
    ) -> None:
        self.backend_name = backend_name
        self.simulator_fallback = simulator_fallback
        self._service = None

    def __repr__(self) -> str:  # noqa: D105
        return f"BackendManager(backend_name={self.backend_name!r})"

    def _try_ibm_runtime_service(self) -> Optional[Any]:
        """Attempt to load qiskit_ibm_runtime.QiskitRuntimeService from saved
        account credentials. Returns None if unavailable (no error raised),
        so callers can gracefully fall back to local simulation."""
        if self._service is not None:
            return self._service
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            self._service = QiskitRuntimeService()
            return self._service
        except Exception:
            return None

    def get_backend(self) -> Backend:
        """Return the target IBM Quantum backend if credentials are
        available, otherwise a local Aer simulator fallback.

        Returns:
            A Qiskit Backend object.
        """
        service = self._try_ibm_runtime_service()
        if service is not None:
            try:
                return service.backend(self.backend_name)
            except Exception as exc:  # pragma: no cover - depends on live creds
                warnings.warn(
                    f"Could not retrieve backend '{self.backend_name}' from "
                    f"IBM Quantum ({exc}); falling back to local simulator."
                )
        if self.simulator_fallback == "aer_noisy":
            return AerSimulator()
        return AerSimulator(method="statevector")

    def transpile_fixed_layout(
        self, circuit: QuantumCircuit, optimization_level: int = 3, seed_transpiler: int = 100000
    ) -> QuantumCircuit:
        """Transpile with fixed-layout, multi-seed selection targeting a
        connected heavy-hex subgraph with zero SWAP insertions (Appendix A.2).

        Args:
            circuit: logical circuit to transpile.
            optimization_level: Qiskit transpiler optimisation level (paper
                uses level 3).
            seed_transpiler: seed for reproducible layout/routing selection.

        Returns:
            Transpiled ISA-compatible QuantumCircuit.
        """
        backend = self.get_backend()
        return transpile(
            circuit,
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=seed_transpiler,
        )

    def run_with_qctrl(
        self, circuit: QuantumCircuit, shots: int, allow_replay_fallback: bool = True
    ) -> Dict[str, int]:
        """Execute a circuit via Q-CTRL's Performance Management Qiskit
        Function (Appendix A.2), or fall back to a documented replay model.

        Args:
            circuit: abstract (untranspiled) circuit to submit -- Q-CTRL's
                managed workflow handles ISA mapping, pulse scheduling, and
                error suppression internally and is not reproducible here.
            shots: number of shots to request.
            allow_replay_fallback: if True and Q-CTRL credentials are not
                configured, raises HardwareUnavailableError with guidance
                instead of silently returning fabricated data; the caller
                should instead use
                `contrast_calibration.ContrastCalibrator.build_hardware_replay_model`
                with the paper's published Tables 17-18 to reproduce the
                *reported statistics* without live hardware access.

        Returns:
            Measurement counts dict, e.g. {"111": 42, "000": 214, ...}.

        Raises:
            HardwareUnavailableError: if no Q-CTRL / IBM Quantum credentials
                are configured (this is the expected path for most users
                reproducing this paper without BasQ/IBM Quantum access).
        """
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService  # noqa: F401
        except ImportError as exc:
            raise HardwareUnavailableError(
                "qiskit-ibm-runtime is not installed; cannot access Q-CTRL "
                "Performance Management. Install it or use the hardware-replay "
                "fallback in estimation/contrast_calibration.py with the "
                "paper's published calibration data (Tables 17-18)."
            ) from exc

        service = self._try_ibm_runtime_service()
        if service is None:
            if allow_replay_fallback:
                raise HardwareUnavailableError(
                    "No IBM Quantum Runtime credentials found. Q-CTRL's "
                    "Performance Management workflow is a managed, proprietary "
                    "service whose internals this repository cannot "
                    "reimplement (see architecture_plan.json risk_assessment). "
                    "To reproduce the paper's reported hardware-replay "
                    "statistics without live hardware access, use "
                    "ContrastCalibrator.build_hardware_replay_model() with the "
                    "calibration parameters from Tables 17-18 instead."
                )
            raise HardwareUnavailableError("No IBM Quantum Runtime credentials configured.")

        # NOTE: the actual Q-CTRL Qiskit Function invocation API
        # (service.run("q-ctrl/...")) requires a live subscription and is
        # intentionally not hard-coded here; consult IBM Quantum's
        # documentation (cited in the paper, IBM Quantum & Q-CTRL, 2026) for
        # the exact function signature available to your account.
        raise NotImplementedError(
            "Live Q-CTRL Performance Management invocation requires "
            "account-specific Qiskit Function access; see docstring."
        )
