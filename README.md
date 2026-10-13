# Bike demand: from laptop model to a model a business can use

ReDi School Hamburg, ML course, session 9: Databricks, MLflow and the ML workflow.

**The task.** A bike-rental company needs to know how many bikes will be rented each hour,
so it can move bikes to the right stations. You build that forecast, keep a record of
everything you try, pick a winner, and update it when new data arrives.

**The data.** Hourly bike rentals from Capital Bikeshare (Washington DC, 2011-2012), with
weather, hour, weekday and season. 17,379 rows. Already in `data/bike_sharing_hourly.csv`.
Original dataset: Fanaee-T and Gama (2013), UCI Machine Learning Repository, CC BY 4.0; downloaded via OpenML.

## Setup (once)

1. **Your own copy of this repo.** On GitHub, click **Use this template** → **Create a new repository**.
   Give it a name, keep it public, click **Create repository**. Copy its URL.
2. **Databricks account.** Search for **Databricks Free Edition** and sign up.
3. **Your repo inside Databricks.** Left menu **Workspace** → your user folder → **Create** → **Git folder**.
   Paste your repo URL, click **Create Git folder**.
4. **Check it runs.** Open `1_train.py`, click **Run**. After a minute you should see a line starting with `run ...`.

## Your steps in the session

1. **Make the training script record itself.** Fill in the three TODOs in `1_train.py`. Run it.
   Open **Experiments** (left menu) → `bike-demand`: your run should show settings and scores.
2. **Experiment.** Change the settings at the top of `1_train.py` and run again. Try the table below,
   then your own ideas. Select runs on the Experiments page and click **Compare**.
3. **Pick a winner.** Run `2_register.py`. It registers your best model and labels it `champion`.
4. **Use it.** Run `3_predict.py`. It loads "the champion" by name and forecasts a June working day.
5. **A new month arrives.** Run `4_new_month.py`. It retrains on newer data and moves the
   `champion` label only if the new model is better.
6. **Save your work to git.** In the Git folder, click the branch name at the top → write a message → **Commit & Push**.

## Things to try in step 2

| Run | MODEL | FEATURES | Other |
|---|---|---|---|
| 1 | `linear` | `raw` | |
| 2 | `linear` | `hour_onehot` | |
| 3 | `forest` | `raw` | `MAX_DEPTH = 5` |
| 4 | `forest` | `raw` | `MAX_DEPTH = 15` |
| 5 | `forest` | `hour_onehot` | `MAX_DEPTH = 15` |
| 6+ | your choice | `mine` | write your own features in `bikes.py`, look for `"mine"` |

## The files

| File | What it does |
|---|---|
| `1_train.py` | Trains one model and logs it to MLflow as one run. **You edit this one.** |
| `2_register.py` | Finds the run with the lowest error, registers its model, labels it `champion`. |
| `3_predict.py` | Loads "the champion" by name and forecasts a June working day. |
| `4_new_month.py` | July data arrives. Retrains, compares with the champion, moves the label if the new one is better. |
| `bikes.py` | Shared helpers: data, features, time split. Optional: your own features under `"mine"`. |
| `solution/1_train.py` | The finished `1_train.py`, if you get stuck. |

## Read next

Chip Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022):
- Chapter 1: why ML in a business is different from ML in a course
- Chapter 6: model development, offline evaluation, and experiment tracking
- Chapters 8-9: why models go stale and how teams retrain them
