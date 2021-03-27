from collections import namedtuple
from typing import Optional

import mlflow
import pandas as pd
from sklearn.metrics import roc_auc_score

from process_and_ml.ml_flow_control import create_experiment
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


def update_weights(x, y, d19):
    ...


class CatPipeline:

    def __init__(self, df: pd.DataFrame, label: bool):
        self.df = df
        self.label = label
        self.features = None
        self.labels = None
        self.trained_model = False
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

        return self.evaluate(lnr, test_df_cleaned)

    def train(self, model, params, tags, run_name: Optional[str], experiment_name: str):
        self.model = model(**params)

        experiment_id = create_experiment(experiment_name=experiment_name)

        with mlflow.start_run(tags=tags, run_name=run_name, experiment_id=experiment_id):
            mlflow.log_params(params)
            self.model.fit(self.X_train, self.y_train, eval_set=(self.X_valid, self.y_valid), verbose=False)
            self.metrics_returned = show_metrics_baseline(self.model, features=self.features, labels=self.labels)
            mlflow.log_metrics(self.metrics_returned)

    def evaluate(self, label, pred):
        prediction = self.model.predict_proba(pred)
        acc = self.model.predict(pred)

        roc_pred = roc_auc_score(label, prediction[:1])

        print(f'Prediction accuracy = {acc}')
        print(f'Prediction roc = {roc_pred} !')
        return prediction[:1], roc_pred, acc
