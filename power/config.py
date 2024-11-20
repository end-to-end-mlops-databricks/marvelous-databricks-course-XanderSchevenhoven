from typing import Any, Dict, List

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    catalog_name: str
    schema_name: str
    model_name: str
    model_artifact_name: str
    experiment_name: str
    numeric_features: List[str]
    target: str
    parameters: Dict[str, Any]  # Dictionary to hold model-related parameters
    ab_test: Dict[str, Any]  # Dictionary to hold A/B test parameters
    package_path: str # path to .whl file to install

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
