from setuptools import find_packages, setup

setup(
    name="sig_vol_id",
    version="0.1.0",
    description=(
        "ArXivist reproduction of 'Signature-based identification of volatility "
        "models from path geometry' (arXiv:2607.06340)"
    ),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26,<2",
        "torch>=2.2,<3",
        "iisignature>=0.24",
        "xgboost>=2.0,<3",
        "scikit-learn>=1.4,<2",
        "pandas>=2.2,<3",
        "matplotlib>=3.8,<4",
        "pyyaml>=6.0,<7",
        "tqdm>=4.66,<5",
    ],
)
