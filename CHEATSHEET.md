# MLflow cheat sheet

Today you need only these commands. Always start your file with `import mlflow`.

## 1. Where your runs go

| Command | What it does | Example |
|---|---|---|
| `mlflow.set_experiment(path)` | Chooses the folder (the "experiment") where your runs are saved. Put it once, at the top. | `mlflow.set_experiment("/Users/you@example.com/bike-demand")` |

Use **your** Databricks email in the path.

## 2. One run

| Command | What it does | Example |
|---|---|---|
| `with mlflow.start_run():` | Starts a run. Everything indented under it belongs to this one run. | see below |
| `with mlflow.start_run(run_name="..."):` | Same, and gives the run a name you can read. | `with mlflow.start_run(run_name="tree depth 5"):` |

```python
with mlflow.start_run(run_name="my first run"):
    # your normal sklearn code goes here, indented
    ...
```

## 3. Record settings (parameters)

A parameter is a choice you made: the model type, the depth, which features.

| Command | What you give it | Example |
|---|---|---|
| `mlflow.log_param(name, value)` | one name and one value | `mlflow.log_param("max_depth", 5)` |
| `mlflow.log_params(dictionary)` | a dictionary: many names and values at once | see below |

```python
params = {"model": "decision tree", "max_depth": 5}   # 1. make a dictionary of your settings
mlflow.log_params(params)                              # 2. give it to MLflow
```

## 4. Record scores (metrics)

A metric is a number that says how good the model is, for example the MAE.

| Command | What you give it | Example |
|---|---|---|
| `mlflow.log_metric(name, value)` | one name and one number | `mlflow.log_metric("mae", mae)` |
| `mlflow.log_metrics(dictionary)` | a dictionary of names and numbers | `mlflow.log_metrics({"mae": mae, "rmse": rmse})` |

## 5. Save the trained model

| Command | What you give it | Example |
|---|---|---|
| `mlflow.sklearn.log_model(model, "model")` | your trained sklearn model, and a folder name | `mlflow.sklearn.log_model(model, "model")` |

## Everything together

```python
import mlflow

mlflow.set_experiment("/Users/you@example.com/bike-demand")

with mlflow.start_run(run_name="tree depth 5"):
    model = ...                    # your sklearn model
    model.fit(...)
    mae = ...                      # your score

    params = {"model": "decision tree", "max_depth": 5}
    mlflow.log_params(params)
    mlflow.log_metrics({"mae": mae})
    mlflow.sklearn.log_model(model, "model")
```

## Where to see it

After a run, the output under your code shows a link **View run … at: https://…**. Click it.
Or: left menu **Experiments** → `bike-demand` → click a run.
