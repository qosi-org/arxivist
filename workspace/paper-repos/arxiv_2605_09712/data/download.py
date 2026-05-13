"""
data/download.py
================
Download FRED-QD panel data and SPF forecasts.

Usage:
    python data/download.py --fred-api-key YOUR_KEY --output-dir data/

Paper: Section 3 — Data and Forecasting Setup
"Quantifying the Risk-Return Tradeoff in Forecasting" (arXiv: 2605.09712)

FRED-QD:
  McCracken, M.W. and Ng, S. (2020). "FRED-QD: A Quarterly Database for
  Macroeconomic Research." Federal Reserve Bank of St. Louis.
  URL: https://research.stlouisfed.org/econ/mccracken/fred-databases/

SPF:
  Survey of Professional Forecasters, Federal Reserve Bank of Philadelphia.
  URL: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters
"""

import argparse
import os
import sys
from pathlib import Path


def download_fred_qd(output_dir: str, api_key: str = "") -> None:
    """
    Download FRED-QD quarterly database.

    The FRED-QD file is available directly from the St. Louis Fed website.
    Transformation codes are in a separate file and must be applied to achieve
    stationarity (see McCracken & Ng 2020, Table 1).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fred_qd_path = output_dir / "FRED-QD.csv"
    tcodes_path = output_dir / "FRED-QD_tcodes.csv"

    if fred_qd_path.exists():
        print(f"FRED-QD already exists at {fred_qd_path}. Skipping download.")
    else:
        print("Downloading FRED-QD...")
        try:
            import urllib.request
            url = "https://files.stlouisfed.org/files/htdocs/fred-md/quarterly/current.csv"
            urllib.request.urlretrieve(url, fred_qd_path)
            print(f"  Saved to {fred_qd_path}")
        except Exception as e:
            print(f"  ERROR downloading FRED-QD: {e}")
            print("  Please download manually from:")
            print("  https://research.stlouisfed.org/econ/mccracken/fred-databases/")
            sys.exit(1)

    if not tcodes_path.exists():
        print("Note: Transformation codes are embedded in row 2 of FRED-QD.csv")
        print("The FREDQDLoader will parse them automatically.")


def download_spf(output_dir: str) -> None:
    """
    Download SPF median forecasts from Philadelphia Fed.

    The SPF provides median probability forecasts for GDP, CPI, unemployment, etc.
    Visit: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/
           survey-of-professional-forecasters
    """
    output_dir = Path(output_dir)
    spf_path = output_dir / "spf_median.csv"

    if spf_path.exists():
        print(f"SPF data already exists at {spf_path}. Skipping.")
        return

    print("SPF data must be downloaded manually from the Philadelphia Fed:")
    print("  https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/survey-of-professional-forecasters")
    print(f"  Save the median forecast file to: {spf_path}")
    print("")
    print("Required series: RGDP, CPI, UNEMP, HOUSING (point forecasts, h=1,2,4)")

    # Create placeholder
    spf_path.write_text(
        "# SPF placeholder — download from Philadelphia Fed\n"
        "# Required columns: date, target, horizon, spf_median\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Download FRED-QD and SPF data")
    parser.add_argument("--fred-api-key", default="", help="FRED API key (optional)")
    parser.add_argument("--output-dir", default="data/", help="Output directory")
    args = parser.parse_args()

    print("=" * 50)
    print("forecast_risk — Data Downloader")
    print("=" * 50)

    download_fred_qd(args.output_dir, args.fred_api_key)
    download_spf(args.output_dir)

    print("\nDone. Verify data files in:", args.output_dir)


if __name__ == "__main__":
    main()
