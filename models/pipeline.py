import pandas as pd

from collections import namedtuple
from typing import List, Tuple, Union, Optional
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from joblib import dump, load

from .constants import PATH_MODELS, RANDOM_STATE

Features = namedtuple('Features', 'X_train X_test X_valid')
Labels = namedtuple('Labels', 'y_train y_test y_valid')
Metrics = namedtuple('Metrics', 'ACC AUC')


class DataSplitsUnitException(Exception):
    """Custom exception to make
    `sklearn.model_selection.train_test_split`
    only works with the float unit
    """
    pass


class DataSplitsSizeException(Exception):
    """Customized exception just to clarify "ValueError"
    from `sklearn.model_selection.train_test_split`
    behavior when the test or validation size is not correct
    and none is equal to zero
    """
    pass


def cat_features_fill_na(df: pd.DataFrame,
                         cat_features: List[str]) -> pd.DataFrame:
    """Fills NA values for each column in `cat_features` for
    `df` dataframe
    """
    df_copy = df.copy()
    # check types

    for cat in cat_features:
        try:
            df_copy[cat] = (
                df_copy[cat].cat.add_categories('UNKNOWN').fillna('UNKNOWN')
            )

        except AttributeError:
            # The dtype is object instead of category
            df_copy[cat] = df_copy[cat].fillna('UNKNOWN')

        # if df[cat]

    return df_copy


def preprocessing_baseline(df: pd.DataFrame,
                           target: str,
                           test_size: float = .15,
                           valid_size: float = .15,
                           cat_features: Optional[List[str]] = None) -> Tuple[Features, Labels]:
    """Creates `features` and `labels` splits and fill NA values
    for categorical features passed in `cat_features` from data
    in `df` dataframe
    Target feature must be provided in `target` arg
    `test_size` and `valid_size` has to be greater than zero and
    less too one, if it is 0 removes that split set
    """
    if 0 < test_size >= 1 or 0 < valid_size >= 1:
        raise DataSplitsUnitException(
            'The parameters test_size and valid_size have to be '
            'greater than zero and less too one'
        )

    x = df.drop(columns=target)
    y = df[target]

    if cat_features:
        x_filled = cat_features_fill_na(x, cat_features=cat_features)
    else:
        x_filled = x.copy()

    try:
        x_train, x_test_and_valid, y_train, y_test_and_valid = (
            train_test_split(
                x_filled,
                y,
                test_size=test_size + valid_size,
                random_state=RANDOM_STATE,
                stratify=y
            )
        )

        x_test, x_valid, y_test, y_valid = (
            train_test_split(x_test_and_valid,
                             y_test_and_valid,
                             test_size=valid_size / (test_size + valid_size),
                             random_state=RANDOM_STATE,
                             stratify=y_test_and_valid)
        )
    except ValueError as value_error:
        if (test_size + valid_size) >= 1:
            raise DataSplitsSizeException(
                'The size of the test and validation data added together '
                'is greater than or equal to one'
            ) from value_error
        elif test_size == valid_size == 0:
            x_train, y_train = x_filled.copy(), y.copy()
            x_test, y_test = pd.DataFrame(), pd.Series()
            x_valid, y_valid = pd.DataFrame(), pd.Series()
        elif test_size == 0:
            x_train, x_valid, y_train, y_valid = train_test_split(
                x_filled,
                y,
                test_size=valid_size,
                random_state=RANDOM_STATE,
                stratify=y
            )

            x_test, y_test = pd.DataFrame(), pd.Series()
        elif valid_size == 0:
            x_train, x_test, y_train, y_test = train_test_split(
                x_filled,
                y,
                test_size=test_size,
                random_state=RANDOM_STATE,
                stratify=y
            )

            x_valid, y_valid = pd.DataFrame(), pd.Series()
        else:
            raise value_error

    return (Features(x_train, x_test, x_valid),
            Labels(y_train, y_test, y_valid))


def compute_metrics(model: Union[Pipeline, CatBoostClassifier],
                    x: pd.DataFrame,
                    y: pd.Series) -> Metrics:
    """Computes `model` metrics for `X` and
    `y`
    """
    predict = model.predict(x)
    predict_proba = model.predict_proba(x)[:, 1]

    acc = accuracy_score(y, predict)
    auc = roc_auc_score(y, predict_proba)

    return Metrics(ACC=acc, AUC=auc)


def show_metrics_baseline(model: Union[Pipeline, CatBoostClassifier],
                          features: Features,
                          labels: Labels) -> None:
    """Giving `model`, `features` and `labels` show accuracy and AUC
    for training, testing and validation data
    Model passed in argument `model` has to be already fitted
    """
    split_names = [field.replace('X_', '').capitalize()
                   for field in features._fields]

    for split_name, split_features, split_labels in zip(split_names,
                                                        features,
                                                        labels):
        if split_features.empty:
            continue

        split_acc, split_auc = compute_metrics(model,
                                               x=split_features,
                                               y=split_labels)

        print(f'Accuracy {split_name}: {split_acc}')
        print(f'AUC {split_name}: {split_auc}')


def target_stats_by_feature(df: pd.DataFrame,
                            feature: str,
                            target: str,
                            fill_na_value: Union[str,
                                                 float] = None) -> pd.DataFrame:
    """Computes the mean and the volume of `target` for each value of `feature`
    """
    df_copy = (
        df.loc[:, [feature, target]].fillna(fill_na_value) if fill_na_value
        else df.loc[:, [feature, target]]
    )

    df_grouped = (
        df_copy
            .groupby(feature)[target]
            .agg(['mean', 'count'])
            .reset_index()
    )

    df_grouped.columns = [feature, f'{target}_mean', f'{target}_count']

    return df_grouped.sort_values(by=f'{target}_mean', ascending=False)


def save_catboost_model(catboost_model: CatBoostClassifier,
                        model_name: str,
                        pool_data: Pool) -> None:
    """Saves model `catboost_model` to `PATH_MODELS` with the name
    passed in `model_name`
    `pool_data` contains `Pool` object with features and labels used
    to fit the model and its categorical features
    """
    catboost_model.save_model(str(PATH_MODELS / model_name), pool=pool_data)


def load_catboost_model(model_name: str) -> CatBoostClassifier:
    """Reads `model_name` from `PATH_MODELS` and returns
    the fitted catboost model
    """
    test_model_from_file = CatBoostClassifier()

    test_model_from_file.load_model(str(PATH_MODELS / model_name))

    return test_model_from_file


def save_pipeline(pipeline: Pipeline, model_name: str) -> None:
    """Saves model `pipeline` to `PATH_MODELS` with the name
    passed in `model_name`
    """
    dump(pipeline, PATH_MODELS / model_name)


def load_pipeline(model_name: str) -> CatBoostClassifier:
    """Reads `model_name` from `PATH_MODELS` and returns
    the fitted catboost model
    """
    return load(PATH_MODELS / model_name)
