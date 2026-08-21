import os
import sys

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
ACCURACY_THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "0.9"))
MODEL_NAME = "iris-classifier"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("iris-ci-cd")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run() as run:
    n_estimators = 100
    max_depth = 3
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(clf, "model")

    print(f"accuracy={acc:.4f}")

    if acc < ACCURACY_THRESHOLD:
        print(f"accuracy {acc:.4f} below threshold {ACCURACY_THRESHOLD}, not registering")
        sys.exit(1)

    model_uri = f"runs:/{run.info.run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)

    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "production", result.version)
    print(f"registered {MODEL_NAME} version {result.version} as @production")
