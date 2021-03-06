from json import dump
from typing import Any

from .projetct_paths import PATH_SUBMISSIONS
from ..config import settings
from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
import pandas as pd


def kaggle_submission(column_lnr: pd.Series,
                      y_pred: np.array,
                      submission_filename: str,
                      submission_message: str) -> None:
    """Function to help to submit attempts on kaggle competition

    Args:
        column_lnr (pd.Series): Client_id column.
        y_pred (np.array): Prediction column.
        submission_filename (str): Name of the the file that ll be submitted.
        submission_message (str): Message to tag your submission.

    Returns:
        None
    """
    filepath = PATH_SUBMISSIONS / f'{submission_filename}.csv'
    df_kaggle_submission = pd.DataFrame(dict(LNR=column_lnr, RESPONSE=y_pred))

    df_kaggle_submission.to_csv(filepath, index=False)

    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    print(kaggle_api
          .competition_submit(filepath,
                              message=submission_message,
                              competition=settings.KAGGLE_COMPETITION))


def serialize_object_dump(object_: Any, filename: str) -> None:
    """Dumps `object` in `PATH_OBJECTS / filename`
    """
    dump(object_, settings.PATH_OBJECTS / filename)
