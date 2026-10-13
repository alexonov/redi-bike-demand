# Bike demand: from laptop model to a model a business can use

ReDi School Hamburg, ML course, session 9: Databricks, MLflow and the ML workflow.

**The task.** A bike-rental company needs to know how many bikes will be rented each hour,
so it can move bikes to the right stations. You build that forecast, keep a record of
everything you try, pick a winner, and update it when new data arrives.

**The data.** Hourly bike rentals from Capital Bikeshare (Washington DC, 2011-2012), with
weather, hour, weekday and season. 17,379 rows. Already in `data/bike_sharing_hourly.csv`.

## Setup (once)

1. Create a free account: search for **Databricks Free Edition** and sign up with your email.
2. In Databricks, click **Workspace** (left menu) → your user folder → **Create** → **Git folder**.
3. Paste this repo's URL, click **Create Git folder**.
4. Open `1_train.py` and click **Run**. After a minute you should see a line starting with `run ...`.

## The files

| File | What it does |
|---|---|
| `1_train.py` | Trains one model and logs it to MLflow as one run. Change the settings at the top and run again. |
| `2_register.py` | Finds the run with the lowest error, registers its model, labels it `champion`. |
| `3_predict.py` | Loads "the champion" by name and forecasts a June working day. |
| `4_new_month.py` | July data arrives. Retrains, compares with the champion, moves the label if the new one is better. |
| `bikes.py` | Shared helpers (data, features, time split). No need to change it. |

## Things to try in `1_train.py`

| Run | MODEL | FEATURES | Other |
|---|---|---|---|
| 1 | `linear` | `raw` | |
| 2 | `linear` | `hour_onehot` | |
| 3 | `forest` | `raw` | `MAX_DEPTH = 5` |
| 4 | `forest` | `raw` | `MAX_DEPTH = 15` |
| 5 | `forest` | `hour_onehot` | `MAX_DEPTH = 15` |

Then open **Experiments** (left menu) → `bike-demand`, select your runs, click **Compare**.

## Read next

Chip Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022):
- Chapter 1: why ML in a business is different from ML in a course
- Chapter 6: model development, offline evaluation, and experiment tracking
- Chapters 8-9: why models go stale and how teams retrain them
