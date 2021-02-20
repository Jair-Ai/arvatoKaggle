from kaggle.api.kaggle_api_extended import KaggleApi
import numpy as np
import pandas as pd


def kaggle_submission(column_lnr: pd.Series,
                      y_pred: np.array,
                      submission_filename: str,
                      submission_message: str) -> None:
    """Submits and saves submission data provided
    in `column_lrt` and `y_pred`
    """
    filepath = PATH_SUBMISSIONS / f'{submission_filename}.csv'
    df_kaggle_submission = pd.DataFrame(dict(LNR=column_lnr, RESPONSE=y_pred))

    df_kaggle_submission.to_csv(filepath, index=False)

    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    print(kaggle_api
          .competition_submit(filepath,
                              message=submission_message,
                              competition='udacity-arvato-identify-customers'))


def serialize_object_dump(object_: Any, filename: str) -> None:
    """Dumps `object` in `PATH_OBJECTS / filename`
    """
    dump(object_, PATH_OBJECTS / filename)
