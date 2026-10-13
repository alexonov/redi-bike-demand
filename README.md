# Teacher demo branch — not for students

Students work on `main` (GUIDE.md, CHEATSHEET.md, empty train.py). This branch holds the teacher's material.

| File | Use |
|---|---|
| `train_reference.py` | What a student's `train.py` looks like after the guide, plus three part-11 variants. Expected MAE: linear 122.5, tree depth 5 97.2, tree depth 10 92.0, forest depth 10 88.5 |
| `1_train.py` | Demo: train one model on a richer feature set (`bikes.py`), log it. Settings at the top |
| `2_register.py` | Demo: register the best run's model, label it `champion` |
| `3_predict.py` | Demo: load "the champion" by name, forecast an average June work day |
| `4_new_month.py` | Demo: July arrives; retrain, compare with the champion, move the label if better |
| `bikes.py` | Helpers for the demo scripts |

## Demo flow (≈10 min, on the projector)

1. Run `1_train.py` with `MODEL="forest"`, `FEATURES="hour_onehot"`, `MAX_DEPTH=15` → validation MAE 46.8.
   (Optional: also `linear`/`raw` → 145.1, to show the table of runs.)
2. Run `2_register.py` → version 1 of `workspace.default.bike_demand`, label `champion`.
   Show it: left menu **Catalog** → `workspace` → `default` → **Models** tab → `bike_demand` → version 1 with the `champion` alias.
3. Run `3_predict.py` → table of hours, actual vs predicted.
4. Run `4_new_month.py` → July: champion 53.8 vs retrained 39.9 → label moves to version 2.

Note: the demo uses a different split (validation Apr–Jun 2012) and features than the students' `train.py`
(validation Jan–Jun 2012), so its MAEs are not comparable with the students'.

Data: Capital Bikeshare hourly rentals, Washington DC, 2011–2012 (Fanaee-T and Gama, UCI Machine Learning Repository, CC BY 4.0).
