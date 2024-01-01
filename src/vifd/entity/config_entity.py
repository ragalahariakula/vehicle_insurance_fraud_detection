from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
    
@dataclass(frozen=True)
class DataValidationConfig:
    unzip_data_dir: Path
    STATUS_FILE: str
    root_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    X_train_data_path: Path
    y_train_data_path: Path
    X_test_data_path: Path
    y_test_data_path: Path
    model_name: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    X_test_data_path: Path
    y_test_data_path: Path
    model_path: Path
    metric_file_name: Path
    mlflow_uri: str