"""Reference for the teacher: a student train.py after all guide steps, plus three part-11 variants.

Set your email first.
Verified on Databricks serverless 2026-10-13: linear 122.5, tree depth 5 97.2, tree depth 10 92.0, forest depth 10 88.5.
"""

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

mlflow.set_experiment("/Users/you@example.com/bike-demand")

bikes = pd.read_csv("data/bike_sharing_hourly.csv")
features = ["hour", "weekday", "month", "temp", "humidity", "windspeed"]

# split by time: learn from 2011, check on the first half of 2012
train = bikes[bikes["year"] == 0]
valid = bikes[(bikes["year"] == 1) & (bikes["month"] <= 6)]

candidates = {
    "linear regression": LinearRegression(),
    "tree depth 5": DecisionTreeRegressor(max_depth=5, random_state=42),
    "tree depth 10": DecisionTreeRegressor(max_depth=10, random_state=42),
    "forest depth 10": RandomForestRegressor(max_depth=10, random_state=42),
}

for name, model in candidates.items():
    with mlflow.start_run(run_name=name):
        model.fit(train[features], train["count"])
        predictions = model.predict(valid[features])
        mae = mean_absolute_error(valid["count"], predictions)
        print(name, "MAE:", round(mae, 1))

        params = {"model": name, "features": ", ".join(features)}
        mlflow.log_params(params)
        mlflow.log_metrics({"mae": mae})
        mlflow.sklearn.log_model(model, "model")
