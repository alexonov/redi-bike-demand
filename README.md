# Bike demand: from laptop model to a model a business can use

ReDi School Hamburg, ML course, session 9: Databricks, MLflow and the ML workflow.

**The task.** A bike-rental company needs to know how many bikes will be rented each hour,
so it can move bikes to the right stations. You build that forecast, keep a record of
everything you try, pick a winner, and update it when new data arrives.

**The data.** Hourly bike rentals from Capital Bikeshare (Washington DC, 2011-2012), with
weather, hour, weekday and season. 17,379 rows. Already in `data/bike_sharing_hourly.csv`.
Original dataset: Fanaee-T and Gama (2013), UCI Machine Learning Repository, CC BY 4.0; downloaded via OpenML.

## Start here

1. **Follow [GUIDE.md](GUIDE.md)** step by step.
2. Keep **[CHEATSHEET.md](CHEATSHEET.md)** open: every MLflow command you need today.
3. You write your model in `train.py`. The data is in `data/`.

## Want to go deeper?

Chip Huyen, *Designing Machine Learning Systems* (O'Reilly, 2022):
- Chapter 1: why ML in a business is different from ML in a course
- Chapter 6: model development, offline evaluation, and experiment tracking
- Chapters 8-9: why models go stale and how teams retrain them
