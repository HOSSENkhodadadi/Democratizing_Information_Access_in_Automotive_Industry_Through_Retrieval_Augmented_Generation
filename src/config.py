"""Configuration loader for Automotive RAG System."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    gold_csv: str
    nlq_dataset: str
    results_dir: str


@dataclass
class ModelConfig:
    name: str
    embedding_dim: int
    device: str


@dataclass
class SearchEngineConfig:
    similarity_metric: str
    capacity: int


@dataclass
class TrainingConfig:
    epochs: int
    patience: int
    preferred_total: int
    mode: str
    norm_order: int
    test_size: float
    random_state: int


@dataclass
class HyperparametersConfig:
    batch_sizes: List[int]
    learning_rates: List[float]
    positive_tendencies: List[float]
    margins: List[float]
    regularization_strengths: List[float]


@dataclass
class GenerationConfig:
    model: str
    temperature: float
    top_p: float
    system_prompt: str


@dataclass
class WebappConfig:
    host: str
    port: int
    debug: bool
    sample_queries: int


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    search_engine: SearchEngineConfig
    training: TrainingConfig
    hyperparameters: HyperparametersConfig
    generation: GenerationConfig
    webapp: WebappConfig
    project_root: Path = field(default_factory=Path)

    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert a relative path to absolute path based on project root."""
        return self.project_root / relative_path


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve().parent
    # Go up until we find pyproject.toml or config folder
    while current != current.parent:
        if (current / "pyproject.toml").exists() or (current / "config").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config/config.yaml
        
    Returns:
        Config object with all settings
    """
    project_root = get_project_root()
    
    if config_path is None:
        config_path = project_root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    config = Config(
        data=DataConfig(**raw_config["data"]),
        model=ModelConfig(**raw_config["model"]),
        search_engine=SearchEngineConfig(**raw_config["search_engine"]),
        training=TrainingConfig(**raw_config["training"]),
        hyperparameters=HyperparametersConfig(**raw_config["hyperparameters"]),
        generation=GenerationConfig(**raw_config["generation"]),
        webapp=WebappConfig(**raw_config["webapp"]),
        project_root=project_root,
    )
    
    return config


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_device(config: Optional[Config] = None) -> str:
    """Get the compute device based on config and availability."""
    import torch
    
    if config is None:
        config = get_config()
    
    if config.model.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return config.model.device
