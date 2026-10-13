# Session 9 guide: build a model and keep a record of every try

**The task.** A bike-rental company wants to know how many bikes people will rent each hour,
so it can move bikes to the right stations. You build that model with sklearn, like in the last sessions.
New today: you record every try in **MLflow**, so you, and your team, can see what was tried and what worked.
Everything runs in **Databricks**, in your browser. You install nothing.

Each part ends with a **Check**: what you should see. If your screen looks different, raise your hand.

Keep [CHEATSHEET.md](CHEATSHEET.md) open next to this guide: it has every MLflow command you need today.

| Part | What you do |
|---|---|
| 1 | Accounts |
| 2 | Your own copy of the project on GitHub |
| 3 | Open it in Databricks |
| 4 | Look at the data (optional scratchpad) |
| 5 | Step 1: a model, no MLflow yet |
| 6 | Step 2: turn it into an MLflow run |
| 7 | Step 3: record your settings |
| 8 | Step 4: record your score |
| 9 | Find your runs in Databricks |
| 10 | Step 5: save the model |
| 11 | Try more models and compare |
| 12 | Save your work to GitHub |

---

## Part 1 — Accounts

1. **GitHub:** log in at [github.com](https://github.com).
2. **Databricks:** search for **Databricks Free Edition**, sign up with your email, and log in. It is free, for learning, no credit card.

**Check:** you are logged in to both.

## Part 2 — Your own copy of the project

The project is a **template**: a starting point you copy. Teams start new projects like this.

1. Open **github.com/alexonov/redi-bike-demand**.
2. Click the green button **Use this template** → **Create a new repository**.
3. Repository name: for example `bike-demand`. Keep it **Public** (so Databricks can read it).
4. Click **Create repository**.
5. On your new repo page, click the green **Code** button and copy the **HTTPS** URL.
   It looks like `https://github.com/YOUR-NAME/bike-demand.git`.

**Check:** the page shows **your** GitHub name in the top left, not `alexonov`.

## Part 3 — Open it in Databricks

A **Git folder** is a copy of your GitHub repo inside Databricks.

1. In Databricks, left menu: **Workspace** → **Users** → your email.
2. Top right: **Create** → **Git folder**.
3. Paste your repo URL. Click **Create Git folder**.

**Check:** you see `data`, `train.py`, `GUIDE.md`, `CHEATSHEET.md`.

## Part 4 — Look at the data (optional scratchpad)

A **notebook** is good for looking around: run one small piece, see the result. Use it as a scratchpad.
Your model goes into `train.py`, a normal Python file, so it is clean and runs from top to bottom.

1. In your Git folder: **Create** → **Notebook**. Name it `scratchpad`.
2. In the first cell:
   ```python
   import pandas as pd
   bikes = pd.read_csv("data/bike_sharing_hourly.csv")
   bikes.head()
   ```
3. Run the cell (Shift + Enter).

The data: one row = one hour, Washington DC, 2011–2012.

| Column | Meaning |
|---|---|
| `count` | bikes rented in that hour. **This is what we predict.** |
| `year` | 0 = 2011, 1 = 2012 |
| `month`, `hour`, `weekday` | when |
| `holiday`, `workingday` | True / False |
| `season`, `weather` | text, for example `spring`, `clear`, `rain` |
| `temp`, `feel_temp`, `humidity`, `windspeed` | the weather in numbers |

**Check:** you see a table with these columns.

## Part 5 — Step 1: a model, no MLflow yet

This is your normal sklearn routine. Open `train.py`.

1. Copy these lines into `train.py`. They load the data and split it **by time**: the model learns from 2011
   and is checked on January–June 2012. Like in real life: we predict a future the model has not seen.
   ```python
   import pandas as pd

   bikes = pd.read_csv("data/bike_sharing_hourly.csv")
   features = ["hour", "weekday", "month", "temp", "humidity", "windspeed"]

   # split by time: learn from 2011, check on the first half of 2012
   train = bikes[bikes["year"] == 0]
   valid = bikes[(bikes["year"] == 1) & (bikes["month"] <= 6)]
   ```
2. Now write the rest yourself, like in the last sessions:
   - import `LinearRegression` and `mean_absolute_error`
   - make the model, `fit` it on `train[features]` and `train["count"]`
   - `predict` on `valid[features]`
   - compute the MAE against `valid["count"]` and `print` it
3. Click **Run** at the top. The first run takes about a minute.

**Check:** the output below your code shows an MAE of about **122**: on average the model misses by 122 bikes per hour.

<details><summary>Show the code for step 1</summary>

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

bikes = pd.read_csv("data/bike_sharing_hourly.csv")
features = ["hour", "weekday", "month", "temp", "humidity", "windspeed"]

# split by time: learn from 2011, check on the first half of 2012
train = bikes[bikes["year"] == 0]
valid = bikes[(bikes["year"] == 1) & (bikes["month"] <= 6)]

model = LinearRegression()
model.fit(train[features], train["count"])
predictions = model.predict(valid[features])
mae = mean_absolute_error(valid["count"], predictions)
print("MAE:", mae)
```
</details>

## Part 6 — Step 2: turn it into an MLflow run

A **run** is one training try. An **experiment** is a folder of runs.

1. At the top of `train.py`, add `import mlflow`.
2. Below the imports, choose the experiment (use **your** Databricks email):
   ```python
   mlflow.set_experiment("/Users/you@example.com/bike-demand")
   ```
3. Put the model lines inside a run. Write `with mlflow.start_run(run_name="linear regression"):`
   and **indent** everything from `model = ...` to `print(...)` under it:
   ```python
   with mlflow.start_run(run_name="linear regression"):
       model = LinearRegression()
       model.fit(train[features], train["count"])
       predictions = model.predict(valid[features])
       mae = mean_absolute_error(valid["count"], predictions)
       print("MAE:", mae)
   ```
4. Run the file.

**Check:** the output now also shows a line **View run linear regression at: https://…**.

## Part 7 — Step 3: record your settings

MLflow wants your settings as a **dictionary**: names and values. Two lines, inside the run (indented):

```python
    params = {"model": "linear regression", "features": ", ".join(features)}   # 1. make the dictionary
    mlflow.log_params(params)                                                  # 2. give it to MLflow
```

`", ".join(features)` turns the list of features into one text, so MLflow can store it.

**Check:** run again. No errors.

## Part 8 — Step 4: record your score

Same idea: a dictionary with the score. Inside the run, after the MAE is computed:

```python
    mlflow.log_metrics({"mae": mae})
```

**Check:** run again. No errors.

## Part 9 — Find your runs in Databricks

1. Click the **View run** link in the output. Or: left menu **Experiments** → `bike-demand` → click a run name.
2. On the run page, find **Parameters**: `model` and `features`. Find **Metrics**: `mae`.
3. Go back to the experiment `bike-demand`. You see a table: one row per run.
   Your first runs (from before Part 7) have no parameters or metrics. That's fine.

**Check:** you see your latest run with `mae` about 122.

## Part 10 — Step 5: save the model

Inside the run, as the last line:

```python
    mlflow.sklearn.log_model(model, "model")
```

**Check:** run again. On the run page, open **Artifacts**: there is a folder `model`.
Your model is saved with its settings and score. Anyone in your team could load it later.

<details><summary>Show the full train.py after step 5</summary>

```python
import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

mlflow.set_experiment("/Users/you@example.com/bike-demand")

bikes = pd.read_csv("data/bike_sharing_hourly.csv")
features = ["hour", "weekday", "month", "temp", "humidity", "windspeed"]

# split by time: learn from 2011, check on the first half of 2012
train = bikes[bikes["year"] == 0]
valid = bikes[(bikes["year"] == 1) & (bikes["month"] <= 6)]

with mlflow.start_run(run_name="linear regression"):
    model = LinearRegression()
    model.fit(train[features], train["count"])
    predictions = model.predict(valid[features])
    mae = mean_absolute_error(valid["count"], predictions)
    print("MAE:", mae)

    params = {"model": "linear regression", "features": ", ".join(features)}
    mlflow.log_params(params)
    mlflow.log_metrics({"mae": mae})
    mlflow.sklearn.log_model(model, "model")
```
</details>

## Part 11 — Try more models and compare

Now improve the model. **Change one thing at a time**, run, and look at the experiment.
Each time, change the `run_name` and the `params` dictionary, so you know later what each run was.

Ideas, from easy to harder:

| Idea | What to change |
|---|---|
| A decision tree | `from sklearn.tree import DecisionTreeRegressor`, `DecisionTreeRegressor(max_depth=5)` |
| Deeper or shallower tree | `max_depth` 3, 5, 10, 15. Put `max_depth` in `params` too! |
| A random forest (many trees, answers averaged) | `from sklearn.ensemble import RandomForestRegressor` |
| More features | add `"workingday"`, `"holiday"`, `"feel_temp"` to the `features` list |
| Text columns | `season` and `weather` are text: turn them into 0/1 columns with `pd.get_dummies` first |
| Your own feature | for example a column that is 1 in rush hour (7–9 and 16–19) and 0 otherwise |

**Compare:** left menu **Experiments** → `bike-demand`. Click the `mae` column header to sort.
Tick 2 or more runs → **Compare**: settings and scores side by side.

**Check:** you have several runs, and you can say which one has the lowest `mae` and why you think it won.

## Part 12 — Save your work to GitHub

1. In your Git folder, click the branch name (`main`) at the top.
2. You see your changed files. Write a short message, for example `train with mlflow`.
3. Click **Commit & Push**. If Databricks asks you to connect GitHub, follow the steps it shows.

**Check:** on github.com, your repo shows your new commit.

---

## Problems and fixes

| Problem | Fix |
|---|---|
| No **Use this template** button | You are not logged in to GitHub |
| Git folder: "repository not found" | You pasted the template URL, not your own. Or your repo is private |
| `FileNotFoundError: data/bike_sharing_hourly.csv` | Run the file from inside your Git folder |
| Error about the experiment path | Check the email in `set_experiment`: exactly the one you log in to Databricks with |
| `IndentationError` | All lines inside `with mlflow.start_run(...):` need the same 4 spaces in front |
| `could not convert string to float: 'spring'` | `season` or `weather` is text: use `pd.get_dummies` first, or leave it out |
| Run takes very long | The first start is slow. Wait 2–3 minutes, then refresh the page |

## Words

| Word | Meaning |
|---|---|
| Databricks | A website where a team keeps code, data and computers in one place |
| Git folder | A copy of your GitHub repo inside Databricks |
| Notebook | A file of small code pieces (cells) you run one at a time. Good as a scratchpad |
| MLflow | A logbook for model training |
| Experiment | A folder of runs for one problem |
| Run | One training try |
| Parameter | A setting you chose: model type, depth, features |
| Metric | A score: MAE |
| MAE | Mean absolute error: the average miss, here in bikes per hour |

## Want to go deeper?

Chip Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022). Very practical. Start with:
- **Chapter 1** — ML in a business vs ML in a course
- **Chapter 2** — ML work as a loop you go round many times
- **Chapter 6** — model development and offline evaluation; the section on experiment tracking is today's topic
- **Chapters 8–9** — why models get worse over time (data drift) and how teams retrain them

---
Data: Capital Bikeshare hourly rentals, Washington DC, 2011–2012 (Fanaee-T and Gama, UCI Machine Learning Repository, CC BY 4.0). The bike-rental company in the story is invented.
