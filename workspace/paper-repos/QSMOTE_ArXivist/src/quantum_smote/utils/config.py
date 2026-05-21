"""Configuration loader and validator for Quantum-SMOTE project.

Provides a typed `Config` dataclass with `from_yaml` and `to_dict` helpers.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    raw_csv_path: str
    target_column: str
    drop_columns: list
    test_size: float
    random_state: int

    clustering: Dict[str, Any]
    quantum_smote: Dict[str, Any]
    classifiers: Dict[str, Any]
    evaluation: Dict[str, Any]
    output: Dict[str, Any]
    logging: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Validate top-level keys
        required_top = [
            "data",
            "clustering",
            "quantum_smote",
            "classifiers",
            "evaluation",
            "output",
        ]
        for k in required_top:
            if k not in raw:
                raise KeyError(f"Missing required config section: {k}")

        data = raw.get("data", {})

        return cls(
            raw_csv_path=data.get("raw_csv_path"),
            target_column=data.get("target_column"),
            drop_columns=data.get("drop_columns", []),
            test_size=float(data.get("test_size", 0.2)),
            random_state=int(data.get("random_state", 42)),
            clustering=raw.get("clustering", {}),
            quantum_smote=raw.get("quantum_smote", {}),
            classifiers=raw.get("classifiers", {}),
            evaluation=raw.get("evaluation", {}),
            output=raw.get("output", {}),
            logging=raw.get("logging", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": {
                "raw_csv_path": self.raw_csv_path,
                "target_column": self.target_column,
                "drop_columns": self.drop_columns,
                "test_size": self.test_size,
                "random_state": self.random_state,
            },
            "clustering": self.clustering,
            "quantum_smote": self.quantum_smote,
            "classifiers": self.classifiers,
            "evaluation": self.evaluation,
            "output": self.output,
            "logging": self.logging,
        }
