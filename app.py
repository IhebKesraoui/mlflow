import os
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def main(data_path, alpha=0.5, l1_ratio=0.5):
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Lisez votre ensemble de données
    data = pd.read_csv(data_path)

    # Colonnes sélectionnées
    selected_columns = [
        'Output/UUT1_I_S1', 'Output/UUT1_I_S2', 'Output/UUT1_I_S3', 'Output/UUT1_I_S4',
        'Output/UUT2_I_S1', 'Output/UUT2_I_S2', 'Output/UUT2_I_S3', 'Output/UUT2_I_S4',
        'Output/UUT3_I_S1', 'Output/UUT3_I_S2', 'Output/UUT3_I_S3', 'Output/UUT3_I_S4'
    ]

    # Diviser les données en ensembles d'apprentissage et de test. (split 0.75, 0.25)
    train, test = train_test_split(data, test_size=0.25)

    # Grouper les colonnes par préfixe (UUT1, UUT2, UUT3)
    grouped_columns = {}
    for column in selected_columns:
        prefix = column.split("/")[1].split("_")[0]  # Extraire le préfixe (UUT1, UUT2, UUT3)
        if prefix not in grouped_columns:
            grouped_columns[prefix] = []
        grouped_columns[prefix].append(column)

    # Itérer sur chaque groupe de colonnes
    for prefix, group in grouped_columns.items():
        print(f"Predicting group {prefix}: {group}")

        # Préparer les données
        train_x = train[group[:-1]]  # Toutes les colonnes sauf la dernière
        test_x = test[group[:-1]]
        
        # La sortie prédite sera la dernière colonne du groupe
        target_column = group[-1]
        
        train_y = train[target_column]
        test_y = test[target_column]

        with mlflow.start_run():
            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            lr.fit(train_x, train_y)

            predicted_qualities = lr.predict(test_x)

            (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

            print(f"ElasticNet model (group={prefix}, alpha={alpha}, l1_ratio={l1_ratio}):")
            print(f"  RMSE: {rmse}")
            print(f"  MAE: {mae}")
            print(f"  R2: {r2}")

            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_param("group", prefix)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            mlflow.sklearn.log_model(lr, "model")

if __name__ == "__main__":
    data_path = "data/aaa.csv"  # Spécifiez le chemin vers votre fichier CSV
    alpha = 0.5  # Paramètre alpha pour ElasticNet
    l1_ratio = 0.5  # Paramètre l1_ratio pour ElasticNet

    main(data_path, alpha, l1_ratio)
