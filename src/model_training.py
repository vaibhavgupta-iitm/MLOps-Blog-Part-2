"""
Model training module with MLflow integration.
Used in Part 2 of the MLOps Blog.
"""

import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, max_depth=3, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None

    def train(self, X_train, y_train):
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        logger.info(f"Model trained with max_depth={self.max_depth}")

    def evaluate(self, X_test, y_test, log_to_mlflow=False):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        if log_to_mlflow:
            mlflow.log_metric("accuracy", acc)
            mlflow.log_dict(report, "classification_report.json")

        return {"accuracy": acc, "report": report}

    def save_model(self, path):
        import joblib
        joblib.dump(self.model, path)
