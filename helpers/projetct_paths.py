from pathlib import Path


# PATH_PROJECT will be called from the root of the project or from a subfolder
PATH_PROJECT = (
    Path('../models') if Path('../models').resolve().name == 'arvato_project'
    else Path('..')
)

PATH_DATA = PATH_PROJECT / 'data'
PATH_MODELS = PATH_PROJECT / 'models'
PATH_SUBMISSIONS = PATH_PROJECT / 'data/submissions'
PATH_OBJECTS = PATH_PROJECT / 'objects'
PATH_DATA_WRANGLER = PATH_PROJECT / 'cleaned'
PATH_MLFLOW_TRACKING = PATH_PROJECT / 'mlruns'
SEP = ';'
NA_VALUES = [0, -1, 'X', 'XX']