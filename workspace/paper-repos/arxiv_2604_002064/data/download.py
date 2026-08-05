#!/usr/bin/env python3
"""Data preparation script for this repository.

Both evaluation problems used in the paper (1D Gaussian density, Section 2.3.1 /
4.2; 5D Black-Scholes Put pricing map, Section 4.3) are generated analytically at
runtime -- see `src/noisy_qnn_uat/data/dataset.py` (`GaussianDensityDataset`,
`BlackScholesPutDataset`). There is no external dataset to download and no
proprietary-data dependency (see architecture_plan.json risk_assessment,
Low-severity "all datasets are synthetic").

This script is kept as a no-op entrypoint for interface consistency with the
standard ArXivist repository layout (`data/download.py`), and to make explicit,
for anyone auditing this repo, that no download step is required.
"""

from __future__ import annotations


def main() -> None:
    print(
        "[data/download.py] No download required: this paper's datasets "
        "(Gaussian density grid, Black-Scholes Put pricing grid) are generated "
        "analytically at runtime by src/noisy_qnn_uat/data/dataset.py. "
        "Nothing to do."
    )


if __name__ == "__main__":
    main()
