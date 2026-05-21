#!/usr/bin/env python3
"""
scripts/reproduce_tables.py
============================
Reproduce Tables 6.2–6.12 from arXiv:2605.06677v1
"Extrema, Barrier Options, and Semi-Analytic Leverage Corrections
in Stochastic-Clock Volatility Models"

Usage
-----
  python scripts/reproduce_tables.py --table all
  python scripts/reproduce_tables.py --table 6.2
  python scripts/reproduce_tables.py --table 6.11 --rho-range -0.9 0.9

All results are printed to stdout and saved as CSV in results/.
"""

import sys
import argparse
import time
import json
from pathlib import Path

import numpy as np

# Make sure the package is importable from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stochastic_clock_barriers.models.cir_clock import CIRClock
from stochastic_clock_barriers.models.sq_ou_clock import SquaredOUClock
from stochastic_clock_barriers.pricing.single_barrier import SingleBarrierPricer
from stochastic_clock_barriers.pricing.double_barrier import DoubleBarrierPricer
from stochastic_clock_barriers.pricing.leverage_expansion import LeverageExpansion
from stochastic_clock_barriers.pricing.pade_accelerator import PadeAccelerator
from stochastic_clock_barriers.monte_carlo.mc_pricer import MonteCarloPricer


# ---------------------------------------------------------------------------
# Paper contract and clock parameters  (Table 6.1)
# ---------------------------------------------------------------------------

CONTRACT = dict(S0=100.0, K=100.0, H=130.0, L=70.0, r=0.03, q=0.0)

CIR_REGIME1  = dict(kappa=0.6, theta=0.20, xi=0.4,  v0=0.18)
CIR_REGIME2  = dict(kappa=0.5, theta=0.45, xi=0.6,  v0=0.48)
SQOU_REGIME1 = dict(alpha=0.6, sigma=0.490, Y0=np.sqrt(0.18))
SQOU_REGIME2 = dict(alpha=0.5, sigma=0.671, Y0=np.sqrt(0.48))

MATURITIES = [0.25, 1.0]

# Expected values from paper (for cross-validation)
PAPER_VALUES = {
    "doc_cir_r1": {0.25: (3.8247, 3.8293, 0.0069), 1.0: (6.4521, 6.4582, 0.0116)},
    "uop_cir_r1": {0.25: (2.9873, 2.9842, 0.0056), 1.0: (4.1056, 4.1097, 0.0104)},
    "doc_cir_r2": {0.25: (5.2134, 5.2195, 0.0102), 1.0: (7.8923, 7.8856, 0.0174)},
    "uop_cir_r2": {0.25: (4.6521, 4.6589, 0.0098), 1.0: (5.8234, 5.8337, 0.0159)},
    "doc_sqou_r1":{0.25: (3.8512, 3.8486, 0.0091), 1.0: (6.5234, 6.5339, 0.0162)},
    "doc_sqou_r2":{0.25: (5.3456, 5.3492, 0.0148), 1.0: (8.1234, 8.1406, 0.0245)},
}

PAPER_EXPANSION_COEFFS_CIR_R1_T1 = [6.4521, 1.6468, -0.4123, 0.2845, -0.1523, 0.0892]

PAPER_PADE_T1 = [
    (-0.9, 4.2134, 3.9182, 4.1827, 4.2053),
    (-0.7, 4.9234, 4.2916, 4.8757, 4.9082),
    (-0.5, 5.5923, 5.7391, 5.5612, 5.5838),
    (-0.3, 5.9456, 5.9672, 5.9378, 5.9432),
    ( 0.3, 6.9845, 6.9646, 6.9776, 6.9819),
    ( 0.5, 7.3412, 7.3276, 7.3308, 7.3385),
    ( 0.7, 7.7089, 7.7227, 7.6852, 7.6991),
    ( 0.9, 8.0912, 8.1497, 8.0612, 8.0806),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rel_err(model, ref):
    """Relative error in percent."""
    if abs(ref) < 1e-12:
        return float("nan")
    return abs(model - ref) / abs(ref) * 100.0


def _print_table(title, headers, rows, col_widths=None):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    if col_widths is None:
        col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
                      for i, h in enumerate(headers)]
    header_line = "  ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
    print()


def _save_csv(path: Path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Tables 6.2-6.5: Independent-clock DOC/UOP under CIR
# ---------------------------------------------------------------------------

def table_6_2_to_6_5(results_dir: Path, run_mc: bool = True, seed: int = 42):
    """Reproduce Tables 6.2–6.5: DOC and UOP prices under CIR clock."""

    for regime_name, cir_params in [("Regime 1", CIR_REGIME1), ("Regime 2", CIR_REGIME2)]:
        clock = CIRClock(**cir_params)
        sb    = SingleBarrierPricer(clock)
        mc    = MonteCarloPricer(clock, n_paths=200_000, seed=seed)  # reduced for speed

        for barrier_type, label in [("doc", "DOC"), ("uop", "UOP")]:
            rows = []
            for T in MATURITIES:
                t0 = time.perf_counter()
                if barrier_type == "doc":
                    sa_price = sb.price_doc(
                        CONTRACT["S0"], CONTRACT["K"], CONTRACT["L"],
                        T, CONTRACT["r"], CONTRACT["q"]
                    )
                else:
                    sa_price = sb.price_uop(
                        CONTRACT["S0"], CONTRACT["K"], CONTRACT["H"],
                        T, CONTRACT["r"], CONTRACT["q"]
                    )
                sa_time = time.perf_counter() - t0

                if run_mc:
                    t0 = time.perf_counter()
                    L = CONTRACT["L"] if barrier_type == "doc" else None
                    H = CONTRACT["H"] if barrier_type == "uop" else None
                    mc_res = mc.price(
                        CONTRACT["S0"], CONTRACT["K"], T,
                        CONTRACT["r"], CONTRACT["q"], rho=0.0,
                        L=L, H=H, barrier_type=barrier_type,
                    )
                    mc_time = time.perf_counter() - t0
                    row = (
                        f"{T:.2f}",
                        f"{sa_price:.4f}",
                        f"{mc_res.price:.4f}",
                        f"{mc_res.stderr:.4f}",
                        f"{_rel_err(sa_price, mc_res.price):.2f}%",
                        f"{sa_time:.3f}s",
                    )
                    rows.append(row)
                else:
                    rows.append((f"{T:.2f}", f"{sa_price:.4f}", "n/a", "n/a", "n/a", f"{sa_time:.4f}s"))

            title = f"Table — {label} prices (CIR clock, {regime_name})"
            headers = ["T", "Semi-analytic", "Monte Carlo", "MC s.e.", "Rel. error", "SA time"]
            _print_table(title, headers, rows)
            key = f"{barrier_type}_cir_{'r1' if regime_name=='Regime 1' else 'r2'}"
            if key in PAPER_VALUES:
                print(f"  [Paper values for {key}]:", PAPER_VALUES[key])
            _save_csv(results_dir / f"table_{label.lower()}_cir_{regime_name.replace(' ', '').lower()}.csv",
                      headers, rows)


# ---------------------------------------------------------------------------
# Tables 6.6-6.7: Independent-clock DOC under Squared OU
# ---------------------------------------------------------------------------

def table_6_6_to_6_7(results_dir: Path, run_mc: bool = True, seed: int = 42):
    """Reproduce Tables 6.6–6.7: DOC prices under Squared OU clock."""

    for regime_name, sqou_params in [("Regime 1", SQOU_REGIME1), ("Regime 2", SQOU_REGIME2)]:
        clock = SquaredOUClock(**sqou_params)
        sb    = SingleBarrierPricer(clock)
        mc    = MonteCarloPricer(clock, n_paths=200_000, seed=seed)

        rows = []
        for T in MATURITIES:
            t0 = time.perf_counter()
            sa_price = sb.price_doc(
                CONTRACT["S0"], CONTRACT["K"], CONTRACT["L"],
                T, CONTRACT["r"], CONTRACT["q"]
            )
            sa_time = time.perf_counter() - t0

            if run_mc:
                mc_res = mc.price(
                    CONTRACT["S0"], CONTRACT["K"], T,
                    CONTRACT["r"], CONTRACT["q"], rho=0.0,
                    L=CONTRACT["L"], barrier_type="doc",
                )
                row = (f"{T:.2f}", f"{sa_price:.4f}",
                       f"{mc_res.price:.4f}", f"{mc_res.stderr:.4f}",
                       f"{_rel_err(sa_price, mc_res.price):.2f}%",
                       f"{sa_time:.4f}s")
            else:
                row = (f"{T:.2f}", f"{sa_price:.4f}", "n/a", "n/a", "n/a", f"{sa_time:.4f}s")
            rows.append(row)

        title = f"Table — DOC prices (Squared OU clock, {regime_name})"
        headers = ["T", "Semi-analytic", "Monte Carlo", "MC s.e.", "Rel. error", "SA time"]
        _print_table(title, headers, rows)
        key = f"doc_sqou_{'r1' if regime_name=='Regime 1' else 'r2'}"
        if key in PAPER_VALUES:
            print(f"  [Paper values]:", PAPER_VALUES[key])
        _save_csv(results_dir / f"table_doc_sqou_{regime_name.replace(' ','').lower()}.csv",
                  headers, rows)


# ---------------------------------------------------------------------------
# Table 6.8: First-order correlation correction
# ---------------------------------------------------------------------------

def table_6_8(results_dir: Path, seed: int = 42):
    """Reproduce Table 6.8: first-order leverage correction."""

    clock = CIRClock(**CIR_REGIME1)
    T = 1.0
    contract = {
        "F0": CONTRACT["S0"], "K": CONTRACT["K"],
        "L":  CONTRACT["L"], "T": T,
        "r":  CONTRACT["r"], "q": CONTRACT["q"],
    }
    exp = LeverageExpansion(
        clock, contract, barrier_type="doc",
        n_paths=50_000, n_steps=50, seed=seed,
    )

    print("\n[Table 6.8] Computing first-order coefficient C1...")
    coeffs = exp.compute_coefficients(order=1)
    C0, C1 = coeffs[0], coeffs[1]
    print(f"  C0 = {C0:.4f} (paper: 6.4521)")
    print(f"  C1 = {C1:.4f} (paper: 1.6468)")

    rho_vals = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rows = []
    for rho in rho_vals:
        v_first = C0 - rho * abs(C1) if C1 < 0 else C0 + rho * C1
        v_first = C0 + rho * C1
        rows.append((f"{rho:.1f}", f"{C0:.4f}", f"{rho*C1:.4f}", f"{v_first:.4f}", "—", "—"))

    headers = ["rho", "V0", "rho*C1", "V^rho (1st)", "MC price", "Rel. error"]
    _print_table("Table 6.8 — First-order correlation correction (DOC, CIR, Regime 1, T=1)", headers, rows)
    _save_csv(results_dir / "table_6_8_first_order.csv", headers, rows)
    return coeffs


# ---------------------------------------------------------------------------
# Table 6.9: Expansion coefficients
# ---------------------------------------------------------------------------

def table_6_9(results_dir: Path, seed: int = 42):
    """Reproduce Table 6.9: expansion coefficients up to order 5."""

    print("\n[Table 6.9] Computing rho-expansion coefficients (order 5)...")
    print("  NOTE: full Duhamel MC is expensive; this may take several minutes.")

    clock = CIRClock(**CIR_REGIME1)
    T = 1.0
    contract = {
        "F0": CONTRACT["S0"], "K": CONTRACT["K"],
        "L":  CONTRACT["L"], "T": T,
        "r":  CONTRACT["r"], "q": CONTRACT["q"],
    }
    exp = LeverageExpansion(
        clock, contract, barrier_type="doc",
        n_paths=50_000, n_steps=50, seed=seed,
    )

    t0 = time.perf_counter()
    coeffs = exp.compute_coefficients(order=5)
    total_time = time.perf_counter() - t0

    paper_coeffs = PAPER_EXPANSION_COEFFS_CIR_R1_T1

    rows = []
    for n, (c, p) in enumerate(zip(coeffs, paper_coeffs)):
        rows.append((f"{n}", f"C_{n}", f"{c:.4f}", f"{p:.4f}", f"{_rel_err(c, p):.1f}%"))

    headers = ["Order n", "Coefficient", "Computed", "Paper", "Rel. diff"]
    _print_table(
        "Table 6.9 — Expansion coefficients (DOC, CIR Regime 1, T=1)",
        headers, rows,
    )
    print(f"  Total computation time: {total_time:.1f}s")
    _save_csv(results_dir / "table_6_9_coefficients.csv", headers, rows)
    return coeffs


# ---------------------------------------------------------------------------
# Table 6.10: Taylor truncations at high rho
# ---------------------------------------------------------------------------

def table_6_10(results_dir: Path, coefficients: list[float] = None):
    """Reproduce Table 6.10: Taylor truncations vs Monte Carlo at high |rho|."""

    if coefficients is None:
        print("[Table 6.10] Using paper coefficients for Taylor evaluation.")
        coefficients = PAPER_EXPANSION_COEFFS_CIR_R1_T1

    rho_vals = [-0.9, -0.7, -0.5, 0.5, 0.7, 0.9]
    mc_prices = {r: v for r, v, *_ in PAPER_PADE_T1}   # from paper Table 6.11

    rows = []
    for rho in rho_vals:
        mc = mc_prices.get(rho, float("nan"))
        t1 = sum(c * rho**n for n, c in enumerate(coefficients[:2]))
        t3 = sum(c * rho**n for n, c in enumerate(coefficients[:4]))
        t5 = sum(c * rho**n for n, c in enumerate(coefficients[:6]))
        rows.append((f"{rho:.1f}", f"{mc:.4f}", f"{t1:.4f}", f"{t3:.4f}", f"{t5:.4f}"))

    headers = ["rho", "MC price", "Order 1", "Order 3", "Order 5"]
    _print_table("Table 6.10 — Taylor truncations vs Monte Carlo", headers, rows)
    _save_csv(results_dir / "table_6_10_taylor.csv", headers, rows)


# ---------------------------------------------------------------------------
# Tables 6.11-6.12: Padé vs Taylor vs MC
# ---------------------------------------------------------------------------

def table_6_11_6_12(results_dir: Path, coefficients: list[float] = None):
    """Reproduce Tables 6.11–6.12: Padé acceleration."""

    if coefficients is None:
        print("[Table 6.11] Using paper coefficients for Padé evaluation.")
        coefficients = PAPER_EXPANSION_COEFFS_CIR_R1_T1

    pade22 = PadeAccelerator(coefficients).build(2, 2)
    pade32 = PadeAccelerator(coefficients).build(3, 2)

    print(f"\n  [2/2] Padé poles: {[f'{p:.4f}' for p in np.roots(pade22._denominator[::-1])]}")
    print(f"  [3/2] Padé poles: {[f'{p:.4f}' for p in np.roots(pade32._denominator[::-1])]}")

    rows = []
    for rho, mc, t5, p22, p32 in PAPER_PADE_T1:
        c_t5   = sum(c * rho**n for n, c in enumerate(coefficients))
        c_p22, _ = pade22.evaluate_safe(rho)
        c_p32, _ = pade32.evaluate_safe(rho)
        rows.append((
            f"{rho:.1f}",
            f"{mc:.4f}",
            f"{c_t5:.4f}  ({_rel_err(c_t5, mc):.2f}%)",
            f"{c_p22:.4f}  ({_rel_err(c_p22, mc):.2f}%)",
            f"{c_p32:.4f}  ({_rel_err(c_p32, mc):.2f}%)",
        ))

    headers = ["rho", "MC price", "Taylor O5 (err%)", "[2/2] Padé (err%)", "[3/2] Padé (err%)"]
    _print_table("Table 6.11 — Padé vs Taylor vs Monte Carlo", headers, rows)
    _save_csv(results_dir / "table_6_11_pade.csv", headers, rows)

    # Table 6.12: error reduction summary
    ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9)]
    rows12 = []
    for lo, hi in ranges:
        subset = [(rho, mc, t5, p22, p32) for rho, mc, t5, p22, p32 in PAPER_PADE_T1
                  if lo <= abs(rho) <= hi]
        if not subset:
            continue
        max_t5_err  = max(_rel_err(sum(c * r**n for n, c in enumerate(coefficients)), mc)
                          for r, mc, *_ in subset)
        max_pade_err = max(
            _rel_err(PadeAccelerator(coefficients).build(3, 2).evaluate_safe(r)[0], mc)
            for r, mc, *_ in subset
        )
        improvement = max_t5_err / max_pade_err if max_pade_err > 1e-8 else float("inf")
        rows12.append((
            f"{lo:.1f}–{hi:.1f}",
            f"{max_t5_err:.2f}%",
            f"{max_pade_err:.2f}%",
            f"{improvement:.0f}×",
        ))

    headers12 = ["|rho| range", "Taylor O5 max err", "Best Padé max err", "Improvement"]
    _print_table("Table 6.12 — Error reduction via Padé", headers12, rows12)
    _save_csv(results_dir / "table_6_12_error_reduction.csv", headers12, rows12)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce tables from arXiv:2605.06677v1")
    parser.add_argument("--table", default="all",
                        choices=["6.2", "6.3", "6.4", "6.5", "6.6", "6.7",
                                 "6.8", "6.9", "6.10", "6.11", "6.12", "all"],
                        help="Which table to reproduce")
    parser.add_argument("--output-dir", default="results",
                        help="Directory for output CSV files")
    parser.add_argument("--no-mc", action="store_true",
                        help="Skip Monte Carlo (semi-analytic only, much faster)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_mc = not args.no_mc

    print(f"\narXiv:2605.06677v1 — Table Reproduction Script")
    print(f"Clock library: stochastic_clock_barriers")
    print(f"Output dir: {results_dir.resolve()}")
    print(f"Run MC: {run_mc}  |  Seed: {args.seed}")

    # Store coefficients so downstream tables can reuse them
    coefficients = None

    table = args.table
    if table in ("6.2", "6.3", "6.4", "6.5", "all"):
        table_6_2_to_6_5(results_dir, run_mc=run_mc, seed=args.seed)

    if table in ("6.6", "6.7", "all"):
        table_6_6_to_6_7(results_dir, run_mc=run_mc, seed=args.seed)

    if table in ("6.8", "all"):
        coefficients = table_6_8(results_dir, seed=args.seed)

    if table in ("6.9", "all"):
        coefficients = table_6_9(results_dir, seed=args.seed)

    if table in ("6.10", "all"):
        table_6_10(results_dir, coefficients)

    if table in ("6.11", "6.12", "all"):
        table_6_11_6_12(results_dir, coefficients)

    print("\n[Done] All requested tables reproduced.")


if __name__ == "__main__":
    main()
