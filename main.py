"""
Basic pipeline orchestrator for Part 2.
Demonstrates MLflow integration + model training.
"""

import argparse
import logging
import mlflow
import pandas as pd
from src.model_training import ModelTrainer
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='iris-dvc-pipeline/v1_data.csv')
    parser.add_argument('--max-depth', type=int, default=3)
    args = parser.parse_args()

    # Set MLflow experiment
    mlflow.set_experiment("iris-part2-pipeline")

    with mlflow.start_run():
        mlflow.log_param("max_depth", args.max_depth)

        # Load data
        df = pd.read_csv(args.data_path)
        X = df.drop('species', axis=1)
        y = df['species']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Train model
        trainer = ModelTrainer(max_depth=args.max_depth)
        trainer.train(X_train, y_train)

        # Evaluate
        metrics = trainer.evaluate(X_test, y_test, log_to_mlflow=True)

        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        mlflow.log_metric("test_accuracy", metrics['accuracy'])

        logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
