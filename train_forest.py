"""
Forest Cover Type — baseline with Random Forest vs Extra Trees.

Pipeline
  1. Feature engineering (elevation combos, water distance, aspect sin/cos,
     soil type decoded into ELU climatic / geologic zones, distance sums/diffs).
  2. 5-fold stratified CV on train.csv to compare RF vs ExtraTrees.
  3. Refit the winner on all of train.csv.
  4. Correct for label shift: train is perfectly balanced (2160 per class) but
     the test set is the full dataset, dominated by classes 1 and 2. We estimate
     the test class priors from the model's own predictions on the unlabeled
     test set (EM, Saerens et al. 2002) and reweight the probabilities.
  5. Write submission.csv (Id, Cover_Type).

Usage:  python train_forest.py [--data-dir DIR] [--out submission.csv] [--trees 300]
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# ELU code for each of the 40 soil types (from the dataset description).
ELU = [2702, 2703, 2704, 2705, 2706, 2717, 3501, 3502, 4201, 4703,
       4704, 4744, 4758, 5101, 5151, 6101, 6102, 6731, 7101, 7102,
       7103, 7201, 7202, 7700, 7701, 7702, 7709, 7710, 7745, 7746,
       7755, 7756, 7757, 7790, 8703, 8707, 8708, 8771, 8772, 8776]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[c for c in ("Id", "Cover_Type") if c in df.columns]).copy()

    hh, vh = X["Horizontal_Distance_To_Hydrology"], X["Vertical_Distance_To_Hydrology"]
    road, fire = X["Horizontal_Distance_To_Roadways"], X["Horizontal_Distance_To_Fire_Points"]
    elev = X["Elevation"]

    # Elevation relative to the nearest water
    X["Elev_minus_VH"] = elev - vh
    X["Elev_minus_HH"] = elev - 0.2 * hh
    X["Dist_Hydro"] = np.sqrt(hh ** 2 + vh ** 2)

    # Aspect is circular
    rad = np.deg2rad(X["Aspect"])
    X["Aspect_sin"], X["Aspect_cos"] = np.sin(rad), np.cos(rad)

    # Distance combinations
    X["HF_sum"], X["HF_diff"] = hh + fire, (hh - fire).abs()
    X["HR_sum"], X["HR_diff"] = hh + road, (hh - road).abs()
    X["FR_sum"], X["FR_diff"] = fire + road, (fire - road).abs()

    # Hillshade summary
    X["Hillshade_mean"] = X[["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]].mean(axis=1)

    # Collapse one-hot columns into single categorical codes
    soil_cols = [f"Soil_Type{i}" for i in range(1, 41)]
    wild_cols = [f"Wilderness_Area{i}" for i in range(1, 5)]
    soil_idx = X[soil_cols].values.argmax(axis=1)  # 0..39
    X["Soil_Id"] = soil_idx + 1
    X["Wilderness_Id"] = X[wild_cols].values.argmax(axis=1) + 1
    elu = np.array(ELU)[soil_idx]
    X["Climatic_Zone"] = elu // 1000
    X["Geologic_Zone"] = (elu // 100) % 10

    return X


def estimate_test_priors(proba, train_prior, n_iter=100, tol=1e-8):
    """EM estimate of test class priors under label shift (Saerens et al. 2002)."""
    prior = train_prior.copy()
    for _ in range(n_iter):
        w = proba * (prior / train_prior)
        w /= w.sum(axis=1, keepdims=True)
        new_prior = w.mean(axis=0)
        if np.abs(new_prior - prior).max() < tol:
            break
        prior = new_prior
    return prior


def adjust(proba, train_prior, test_prior):
    p = proba * (test_prior / train_prior)
    return p / p.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/mnt/user-data/uploads")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train = pd.read_csv(f"{args.data_dir}/train.csv")
    test = pd.read_csv(f"{args.data_dir}/test-full.csv")
    X, y = add_features(train), train["Cover_Type"].values
    X_test = add_features(test)
    classes = np.sort(np.unique(y))
    print(f"train {X.shape}, test {X_test.shape}")

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=args.trees, n_jobs=-1, random_state=args.seed),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=args.trees, n_jobs=-1, random_state=args.seed),
    }

    # --- 1. Cross-validation ---------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof = {}
    for name, model in models.items():
        t0 = time.time()
        oof[name] = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
        acc = (classes[oof[name].argmax(1)] == y).mean()
        print(f"{name:13s} CV accuracy (balanced train): {acc:.4f}  [{time.time()-t0:.0f}s]")

    best = max(oof, key=lambda n: (classes[oof[n].argmax(1)] == y).mean())
    print(f"-> selected {best}")

    # --- 2. Refit on all training data, predict test ---------------------
    model = models[best].fit(X, y)
    proba = np.vstack([model.predict_proba(X_test.iloc[i:i + 100_000])
                       for i in range(0, len(X_test), 100_000)])

    # --- 3. Label-shift correction ---------------------------------------
    train_prior = np.bincount(y, minlength=classes.max() + 1)[classes] / len(y)
    test_prior = estimate_test_priors(proba, train_prior)
    print("estimated test priors:",
          {int(c): round(float(p), 3) for c, p in zip(classes, test_prior)})

    # How much would the correction help? Score CV predictions weighted as if
    # the classes appeared with the estimated test frequencies.
    w = (test_prior / train_prior)[np.searchsorted(classes, y)]
    raw_hit = classes[oof[best].argmax(1)] == y
    adj_hit = classes[adjust(oof[best], train_prior, test_prior).argmax(1)] == y
    print(f"expected test accuracy  raw: {np.average(raw_hit, weights=w):.4f}  "
          f"prior-adjusted: {np.average(adj_hit, weights=w):.4f}")

    pred = classes[adjust(proba, train_prior, test_prior).argmax(1)]

    # --- 4. Submission ---------------------------------------------------
    sub = pd.DataFrame({"Id": test["Id"], "Cover_Type": pred})
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(sub)} rows)")
    print("predicted class distribution:",
          sub["Cover_Type"].value_counts(normalize=True).sort_index().round(3).to_dict())


if __name__ == "__main__":
    main()
