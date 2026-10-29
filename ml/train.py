"""Train Logistic Regression, Random Forest, and XGBoost; save the best calibrated model.

Usage:
    python ml/train.py
    python ml/train.py path/to/diabetic_data.csv
    python ml/train.py path/to/mimic_csv_folder
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.common import CATEGORICAL, FEATURES, NUMERIC, load_kaggle
from ml.mimic import load_mimic

warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
OUT = HERE / "artifacts"
OUT.mkdir(exist_ok=True)


def preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])


def candidates(scale_pos_weight):
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


def main():
    source_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data" / "diabetic_data.csv"
    if source_path.is_dir():
        df, source = load_mimic(source_path), f"MIMIC-III v1.4 ({source_path})"
    elif source_path.exists():
        df, source = load_kaggle(source_path), f"Kaggle diabetes 130-US hospitals ({source_path.name})"
    else:
        raise FileNotFoundError(f"Training data not found: {source_path}")

    print("Data source:", source, "| rows:", len(df), "| readmit rate: %.1f%%" % (100 * df.target.mean()))
    X, y = df[FEATURES], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    imbalance = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    results, fitted = {}, {}

    for name, classifier in candidates(imbalance).items():
        pipeline = Pipeline([("preprocess", preprocessor()), ("classifier", classifier)])
        started = time.perf_counter()
        cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1).mean()
        pipeline.fit(X_train, y_train)
        inference_started = time.perf_counter()
        probability = pipeline.predict_proba(X_test)[:, 1]
        inference_ms = (time.perf_counter() - inference_started) * 1000 / len(X_test)
        prediction = (probability >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
        results[name] = {
            "roc_auc": round(roc_auc_score(y_test, probability), 4),
            "pr_auc": round(average_precision_score(y_test, probability), 4),
            "cv_roc_auc": round(float(cv_auc), 4),
            "recall": round(recall_score(y_test, prediction), 4),
            "precision": round(precision_score(y_test, prediction, zero_division=0), 4),
            "f1": round(f1_score(y_test, prediction), 4),
            "brier_score": round(brier_score_loss(y_test, probability), 4),
            "training_seconds": round(time.perf_counter() - started, 2),
            "inference_ms_per_patient": round(inference_ms, 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }
        fitted[name] = pipeline
        print(f"{name:22s}", results[name])

    best = max(results, key=lambda name: results[name]["roc_auc"])
    baseline = {column: float(X_train[column].median()) for column in NUMERIC}
    baseline.update({column: X_train[column].mode()[0] for column in CATEGORICAL})

    print("Calibrating selected model:", best)
    final = CalibratedClassifierCV(clone(fitted[best]), method="sigmoid", cv=3, n_jobs=1).fit(X_train, y_train)
    calibrated = final.predict_proba(X_test)[:, 1]
    low, high = (float(value) for value in np.percentile(calibrated, [50, 85]))
    tiers = {}
    for tier, mask in (
        ("Low", calibrated < low),
        ("Medium", (calibrated >= low) & (calibrated < high)),
        ("High", calibrated >= high),
    ):
        tiers[tier] = {
            "patients": int(mask.sum()),
            "observed_readmit_rate": round(float(y_test[mask].mean()), 4),
        }

    results[best]["calibrated_roc_auc"] = round(roc_auc_score(y_test, calibrated), 4)
    results[best]["calibrated_pr_auc"] = round(average_precision_score(y_test, calibrated), 4)
    results[best]["calibrated_brier_score"] = round(brier_score_loss(y_test, calibrated), 4)
    print("Calibrated tiers (Low<%.3f, High>=%.3f):" % (low, high), tiers)

    joblib.dump({
        "pipeline": final,
        "name": best,
        "baseline": baseline,
        "thresholds": [low, high],
        "features": FEATURES,
    }, OUT / "model.joblib")
    (OUT / "metrics.json").write_text(json.dumps({
        "source": source,
        "best": best,
        "selection_metric": "held-out ROC-AUC",
        "results": results,
        "tiers": tiers,
        "thresholds": [round(low, 4), round(high, 4)],
        "base_rate": round(float(y.mean()), 4),
    }, indent=2))
    print("Best model:", best, "-> saved to", OUT / "model.joblib")


if __name__ == "__main__":
    main()
