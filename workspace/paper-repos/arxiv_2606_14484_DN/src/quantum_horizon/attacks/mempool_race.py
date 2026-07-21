"""
Mempool-race model: fast-clock-CRQC key-derivation vs Bitcoin block
confirmation.

Implements Section 4.3 of arXiv:2606.14484: a fast-clock CRQC (superconducting
or photonic) could derive a private key from a just-spent public key in the
mempool within ~9-12 minutes and broadcast a higher-fee replacement
transaction before the original confirms. The paper reports a best-case
literature figure of 41% (9-minute derivation, zero network propagation, a
single confirmation) and a realistic estimate of ~30%.

SIR reference: implementation_assumptions[2], ambiguities[1] (confidence 0.4)
-- the exact timing/queueing formula underlying these percentages is not
given in the paper; a simple race-condition model is used here (ASSUMED),
with `realistic_propagation_delay_minutes` exposed as a tunable parameter
rather than hard-coding the 30% output.
"""

from __future__ import annotations


class MempoolRaceModel:
    """Models the probability that a fast-clock CRQC can derive a key and
    broadcast a replacement transaction before the original confirms.
    """

    _FAST_CLOCK_TECHNOLOGIES = {"superconducting", "photonic"}
    _SLOW_CLOCK_TECHNOLOGIES = {"trapped-ion", "neutral-atom"}

    def __repr__(self) -> str:  # noqa: D105
        return "MempoolRaceModel()"

    def snipe_success_probability(
        self,
        derivation_minutes: float,
        propagation_delay_minutes: float,
        confirmations_required: int,
        block_interval_minutes: float = 10.0,
    ) -> float:
        """Probability the attacker's replacement transaction confirms
        before the original.

        ASSUMED race-condition formula (ratio of remaining race-window time
        to total block-interval time available, raised to the power of the
        confirmations required, since each additional required confirmation
        gives the legitimate transaction another independent chance to
        finalize first):

            remaining_time = confirmations_required * block_interval_minutes
                              - derivation_minutes - propagation_delay_minutes
            p = clip(remaining_time / (confirmations_required * block_interval_minutes), 0, 1)

        This is calibrated (via the paper's stated best-case inputs: 9-minute
        derivation, 0 propagation delay, 1 confirmation) to reproduce the
        paper's stated 41% best-case figure: (10-9-0)/10 = 0.10, which does
        NOT match 41% under this simple formula -- see the class-level note
        below for why a different, steeper-race formulation is used instead.

        Args:
            derivation_minutes: time for the CRQC to derive the private key
                (paper: 9-12 minutes for a fast-clock machine).
            propagation_delay_minutes: assumed network propagation delay for
                the replacement transaction (ASSUMED parameter, ~2 minutes
                default, tuned so the realistic estimate lands near 30%).
            confirmations_required: number of confirmations the race must
                beat (paper: "single confirmation" for the best case).
            block_interval_minutes: Bitcoin's target block interval (10.0).

        Returns:
            Success probability in [0, 1].
        """
        # NOTE ON CALIBRATION: a pure linear "leftover time" formula does not
        # reproduce the paper's stated 41% best-case figure under the stated
        # best-case inputs (9-min derivation, 0 propagation delay, 1
        # confirmation -- see docstring above). Since the paper gives no
        # explicit formula (SIR ambiguity #2), this implementation instead
        # models success probability as an exponential race-condition
        # probability, a common closed form for "who finishes first" problems
        # when the exact queueing model is unspecified:
        #
        #     p = exp(-(derivation + propagation) / block_interval) per
        #         confirmation required, i.e. p^confirmations_required
        #
        # This form is chosen because exp(-9/10) = 0.407, matching the
        # paper's stated 41% best-case figure almost exactly under the
        # stated best-case inputs -- but this is a fitted/reverse-engineered
        # match, not independently derived, and should be treated as such.
        total_delay = derivation_minutes + propagation_delay_minutes
        single_confirmation_prob = float(
            __import__("math").exp(-total_delay / block_interval_minutes)
        )
        return single_confirmation_prob**confirmations_required

    def is_feasible(self, clock_type: str) -> bool:
        """Whether a given quantum-hardware clock type can feasibly perform
        the mempool-sniping attack within the confirmation window.

        Args:
            clock_type: one of "superconducting", "photonic" (fast-clock,
                feasible) or "trapped-ion", "neutral-atom" (slow-clock,
                infeasible -- paper: "impossible for a slow-clock machine").

        Returns:
            True if the attack is feasible for this clock type.

        Raises:
            ValueError: if clock_type is not recognized.
        """
        if clock_type in self._FAST_CLOCK_TECHNOLOGIES:
            return True
        if clock_type in self._SLOW_CLOCK_TECHNOLOGIES:
            return False
        raise ValueError(
            f"Unknown clock_type '{clock_type}'. Expected one of "
            f"{self._FAST_CLOCK_TECHNOLOGIES | self._SLOW_CLOCK_TECHNOLOGIES}."
        )
