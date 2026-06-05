"""
Script to run multiple experiments using MLflow.
This demonstrates hyperparameter comparison for the IRIS classification model.
Part of MLOps Blog Part 2.
"""

import argparse
import logging
import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_path: str):
    df = pd.read_csv(data_path)
    X = df.drop('species', axis=1)
    y = df['species']
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def run_single_experiment(X_train, X_test, y_train, y_test, params: dict, experiment_name: str):
    with mlflow.start_run(run_name=params.get('name', 'experiment')):
        mlflow.log_params(params)
        
        model = DecisionTreeClassifier(
            max_depth=params.get('max_depth', 3),
            min_samples_split=params.get('min_samples_split', 2),
            criterion=params.get('criterion', 'gini'),
            random_state=42
        )
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_score", test_f1)
        
        # Log model
        signature = infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="iris-classifier"
        )
        
        logger.info(f"Experiment {params['name']}: Test Acc = {test_acc:.4f}")
        return test_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='iris-dvc-pipeline/v1_data.csv')
    parser.add_argument('--experiment-name', type=str, default='iris-experiments-part2')
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)
    X_train, X_test, y_train, y_test = load_data(args.data_path)

    experiments = [
        {"name": "shallow_tree", "max_depth": 2},
        {"name": "medium_tree", "max_depth": 4},
        {"name": "deep_tree", "max_depth": 8},
        {"name": "entropy_criterion", "max_depth": 4, "criterion": "entropy"},
    ]

    for exp in experiments:
        run_single_experiment(X_train, X_test, y_train, y_test, exp, args.experiment_name)

    logger.info("All experiments completed. View results in MLflow UI.")

if __name__ == "__main__":
    main()
