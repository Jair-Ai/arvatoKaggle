from pathlib import Path


# PATH_PROJECT will be called from the root of the project or from a subfolder
PATH_PROJECT = (
    Path('.') if Path('.').resolve().name == 'Arvato-Customer-Segmentation'
    else Path('..')
)

PATH_DATA = PATH_PROJECT / 'data'
PATH_FILE_ATTRIBUTES = PATH_DATA / 'feature_attributes.xlsx'
PATH_MODELS = PATH_PROJECT / 'models'
PATH_SUBMISSIONS = PATH_PROJECT / 'submissions'
PATH_OBJECTS = PATH_PROJECT / 'objects'
PATH_MLFLOW_TRACKING = PATH_PROJECT / 'mlruns'
SEP = ';'
NA_VALUES = [0, -1, 'X', 'XX']
RANDOM_STATE = 42