# This code is copied pasted from mlflow wine dataset tutorial

import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    csv_url = ("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv")
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download training & test CSV, check your internet connection. Error: %s", e)

    train, test = train_test_split(data)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    with mlflow.start_run():
        alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5 # argv: positional argument
        l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
        ## run the code: python app.py 0.7 0.3   → alpha = 0.7, l1_ratio = 0.3    '.' len(sys.argv) = 2
        ## run the code: python app.py           → alpha = 0.5, l1_ratio = 0.5    '.' len(sys.argv) = 0

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        # mlflow Tracking
        mlflow.log_param("alpha", alpha) # mlflow Tracking parameters
        mlflow.log_param("l1_ratio", l1_ratio) # mlflow Tracking parameters
        mlflow.log_metric("rmse", rmse) # mlflow Tracking metrics
        mlflow.log_metric("r2", r2) # mlflow Tracking metrics
        mlflow.log_metric("mae", mae) # mlflow Tracking metrics

        predictions = lr.predict(train_x)
        signature = infer_signature(train_x, predictions)

        # For Remote server only (DAGSHUB)
        Remote_Server_URI="https://dagshub.com/anweshabose/Udemy_MLFlow_Experiments_Wine_Data.mlflow"
        mlflow.set_tracking_uri(Remote_Server_URI)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # This url will be useful while running in dagshub remote repository

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel", signature=signature) # log_model: to select the best model
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)


# (d:\Udemy\Complete_DSMLDLNLP_Bootcamp\Python\50-MLFlow Dagshub and BentoML\venv) D:\Udemy\Complete_DSMLDLNLP_Bootcamp\Python\50-MLFlow Dagshub and BentoMagshub and BentoML>python app.py
#Elasticnet model (alpha=0.500000, l1_ratio=0.500000):
#  RMSE: 0.7931640229276851
#  MAE: 0.6271946374319586
#  R2: 0.10862644997792614
# It will also create "mlruns" folder