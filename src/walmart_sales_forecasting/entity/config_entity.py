from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: str
    schema: dict
    data_dirs: dict = field(
        default_factory=lambda: {
            'features': Path,
            'stores': Path,
            'train': Path,
            'test': Path
        }
    )


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_dirs: dict = field(
        default_factory=lambda: {
            'features': Path,
            'stores': Path,
            'train': Path,
            'test': Path
        }
    )


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    pipeline_name: str
    model_instance_name: str
    n_estimators: int
    learning_rate: float
    random_state: int
    n_jobs: int
    target_column: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    pipeline_path: Path
    test_data_path: str
    evaluation_metrics_path: Path
    model_params: dict
    target_column: str
    mlflow_uri: str
    mlflow_project_name: str
