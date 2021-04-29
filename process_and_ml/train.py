from collections import namedtuple
from typing import Optional, Union, Dict, List

import mlflow
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, cross_val_predict, train_test_split

from config import settings
from process_and_ml.ml_flow_control import create_experiment
from process_and_ml.models import compute_metrics
from process_and_ml.pipeline import preprocessing_baseline, cat_features_fill_na, show_metrics_baseline
from pathlib import Path

path = (
    Path('.') if Path('.').resolve().name == 'arvato_project'
    else Path('..')
)
path_mlflow = path / 'mlruns'
mlflow.set_tracking_uri(str(path_mlflow))

Features = namedtuple('Features', 'X_train X_test X_valid')
Labels = namedtuple('Labels', 'y_train y_test y_valid')


def evaluate(label, trained_model, pred=None):
    prediction = trained_model.predict_proba(pred)
    acc = trained_model.predict(pred)

    roc_pred = roc_auc_score(label, prediction[:1])

    print(f'Prediction accuracy = {acc}')
    print(f'Prediction roc = {roc_pred} !')
    return prediction[:1], roc_pred, acc


def log_best(run: mlflow.entities.Run,
             metric: str) -> None:
    """Log the best parameters from optimization to the parent experiment.

    Args:
        run: current run to log metrics
        metric: name of metric to select best and log
    """

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        [run.info.experiment_id],
        "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id))

    best_run = min(runs, key=lambda run: run.data.metrics[metric])

    mlflow.set_tag("best_run", best_run.info.run_id)
    mlflow.log_metric(f"best_{metric}", best_run.data.metrics[metric])


def update_weights(x, y, d19):
    ...


class CatPipeline:

    def __init__(self, df: pd.DataFrame, label: bool):
        self.df = df
        self.label = label
        self.features = None
        self.labels = None
        self.model = None
        self.class_weights = None
        self.X_train = None
        self.X_valid = None
        self.y_train = None
        self.y_valid = None
        self.X_test = None
        self.y_test = None
        self.cat_features = None
        self.metrics_returned = None
        self.non_wrangler_sequence(df)

    def non_wrangler_sequence(self, df: pd.DataFrame):
        self.cat_features = df.select_dtypes(include=['category', 'object']).columns

        self.features, self.labels = preprocessing_baseline(df,
                                                            cat_features=self.cat_features,
                                                            target='is_customer')

        self.X_train, self.X_test, self.X_valid = self.features
        self.y_train, self.y_test, self.y_valid = self.labels

        self.class_weights = (1, sum(self.y_train == 0) / sum(self.y_train == 1))

    def predict_test(self, test_df: pd.DataFrame):
        lnr = test_df['LNR']

        test_df_cleaned = cat_features_fill_na(test_df.drop(columns=['LNR'], errors='ignore'),
                                               cat_features=self.cat_features)

        return evaluate(lnr, test_df_cleaned)

    def train(self, model, params, tags, run_name: Optional[str], experiment_name: str):
        self.model = model(**params)

        experiment_id = create_experiment(experiment_name=experiment_name)

        with mlflow.start_run(tags=tags, run_name=run_name, experiment_id=experiment_id):
            mlflow.log_params(params)
            self.model.fit(self.X_train, self.y_train, eval_set=(self.X_valid, self.y_valid), verbose=False)
            self.metrics_returned = show_metrics_baseline(self.model, features=self.features, labels=self.labels)
            mlflow.log_metrics(self.metrics_returned)


class TrainAfterPipeline:

    def __init__(self, dataset: np.array, label: np.array):
        self.dataset = dataset
        self.label = label
        self.trained_model = False
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset, self.label.values,
                                                                                random_state=settings.RANDOM_STATE,
                                                                                test_size=0.3)

    def train_grid_search(self, model, grid: Dict[str, Union[str, float, List[Union[str, int, float]]]]):
        cv = RepeatedStratifiedKFold(n_splits=20, n_repeats=3, random_state=settings.RANDOM_STATE)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='roc_auc', verbose=2)

        grid_result = grid_search.fit(self.X_train, self.y_train)

        print(grid_result.best_score_)
        print(grid_result.best_estimator_)
        compute_metrics(grid_result.best_estimator_, self.X_test, self.y_test)

        return grid_result

    def train(self, model, params, tags, run_name: Optional[str], experiment_name: str, data: Optional[Dict[str, np.array]]=None):
        model = model(**params)

        experiment_id = create_experiment(experiment_name=experiment_name)

        if not data:
            x = self.X_train
            y = self.y_train
        else:
            x = data['x']
            y = data['y']
        with mlflow.start_run(tags=tags, run_name=run_name, experiment_id=experiment_id):
            mlflow.log_params(params)
            model.fit(x, y)
            split_acc, split_auc = compute_metrics(model, self.X_test, self.y_test)
            mlflow.log_metrics({'acc': split_acc, 'auc': split_auc})

        return model
