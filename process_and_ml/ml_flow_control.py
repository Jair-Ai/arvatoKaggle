import mlflow
from mlflow.exceptions import MlflowException

from helpers.projetct_paths import PATH_MLFLOW_TRACKING


def create_experiment(experiment_name: str):
    """Create or take experiment name

    Args:
        experiment_name: Name for your experiment

    Returns:
        The experiment name.

    """
    mlflow.set_tracking_uri(str(PATH_MLFLOW_TRACKING))
    try:
        return mlflow.create_experiment(experiment_name)
    except MlflowException:
        return mlflow.get_experiment_by_name(experiment_name).experiment_id
