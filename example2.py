import os
import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from dotenv import dotenv_values, load_dotenv

from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    load_dotenv(".secrets")

    print(f"Tracking uri: {os.environ['MLFLOW_TRACKING_URI']}")

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    mlflow.set_experiment("wine_quality_experiment")
    mlflow.autolog()

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(['quality'], axis=1), 
        data['quality'], 
        stratify=data['quality'],
        test_size=0.25
    )

    # Define distribution to pick parameter values from
    distributions = dict(
        alpha=uniform(loc=float(sys.argv[1]), scale=float(sys.argv[2])),  # sample alpha uniformly from [-5.0, 5.0]
        l1_ratio=uniform(),  # sample l1_ratio uniformlyfrom [0, 1.0]
    )

    with mlflow.start_run():
        lr = ElasticNet()

        # Initialize random search instance
        clf = RandomizedSearchCV(
            estimator=lr,
            param_distributions=distributions,
            # Optimize for mean absolute error
            scoring="neg_mean_absolute_error",
            # Use 5-fold cross validation
            cv=5,
            # Try 100 samples. Note that MLflow only logs the top 5 runs.
            n_iter=100,
            verbose=3
        )

        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_

        y_pred = best_model.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, y_pred)
        metric_dict = {
            "validation_rmse": rmse,
            "validation_mae": mae,
            "validation_r2": r2
        }
        mlflow.log_metrics(metric_dict)

        print(f"best_model Elastic net: {best_model}")
        mlflow.log_params(dict(**best_model.get_params()))
        print("Best model metrics:", metric_dict)

        signature = infer_signature(X_train, best_model.predict(X_train))
        print("Signature:", signature, "type:", type(signature))

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                best_model, "explicit_model", registered_model_name="ElasticnetWineModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(best_model, "explicit_model", signature=signature)



