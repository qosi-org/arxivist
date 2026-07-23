from setuptools import find_packages, setup

setup(
    name="ews_kalman",
    version="0.1.0",
    description="Reproduction of arXiv:2607.11935 -- TVP-Kalman eigenvector-rotation early-warning signals",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
