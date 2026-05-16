from setuptools import setup, find_packages

setup(
    name="agenticaita",
    version="0.1.0",
    description="AGENTICAITA: Deliberative Multi-Agent Reasoning for Autonomous Trading (arxiv:2605.12532)",
    author="ArXivist reproduction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "aiohttp>=3.9.0",
        "aiosqlite>=0.19.0",
        "pydantic>=2.5.0",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
        "python-socks>=2.4.0",
        "websockets>=12.0",
        "rich>=13.0",
    ],
)
