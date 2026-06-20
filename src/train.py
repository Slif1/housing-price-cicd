import json

import joblib
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data


def train():
    """train the model via a sklearn pipeline
    Returns:
        metrics: the metrics from the trained model
    """
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(max_depth=4, random_state=42)),
        ]
    )

    with mlflow.start_run():
        model = pipeline.named_steps["model"]
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("random_state", model.random_state)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        metrics = {"rmse": rmse, "r2": r2}

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")

    with open("metrics.json", "w") as f:
        json.dump(metrics, f)
    joblib.dump(pipeline, filename="models/pipeline.pkl")

    return metrics


if __name__ == "__main__":
    metrics = train()
    print(f"RMSE : {metrics['rmse']:.4f}")
    print(f"R²   : {metrics['r2']:.4f}")
