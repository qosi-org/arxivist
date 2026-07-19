from setuptools import setup, find_packages
setup(
    name="red_cohort",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.2", "numpy>=1.26", "networkx>=3.3",
        "scipy>=1.13", "matplotlib>=3.9", "pyyaml>=6.0",
        "tqdm>=4.66", "orjson>=3.10",
    ],
)
