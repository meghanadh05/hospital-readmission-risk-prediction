"""Shared feature definitions + data loading for training and Django prediction.

Preprocessing follows the reference Kaggle notebook "Prediction on hospital readmission":
drop expired patients, keep each patient's FIRST encounter, group ICD-9 diagnosis codes into
9 disease categories, add service_utilization and numchange, regroup admission/discharge ids.
"""
import numpy as np
import pandas as pd

NUMERIC = ["age", "time_in_hospital", "num_lab_procedures", "num_procedures",
           "num_medications", "number_outpatient", "number_emergency",
           "number_inpatient", "number_diagnoses", "service_utilization", "numchange"]
CATEGORICAL = ["gender", "A1Cresult", "max_glu_serum", "change", "diabetesMed",
               "admission_type", "discharge_group", "diag_group"]
FEATURES = NUMERIC + CATEGORICAL

DIAG_GROUPS = ["Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
               "Musculoskeletal", "Genitourinary", "Neoplasms", "Other"]
DRUGS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
         "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
         "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "insulin",
         "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
         "metformin-rosiglitazone", "metformin-pioglitazone"]


def _age_to_num(v):
    if isinstance(v, str) and v.startswith("["):  # '[70-80)' -> 75
        return int(v[1:].split("-")[0]) + 5
    return float(v)


def diag_group(code):
    """ICD-9 code -> one of 9 disease groups (same ranges as the reference notebook)."""
    s = str(code)
    if s[:1] in ("V", "E"):
        return "Other"
    try:
        v = float(s)
    except ValueError:
        return "Other"
    f = int(v)
    if 390 <= v < 460 or f == 785: return "Circulatory"
    if 460 <= v < 520 or f == 786: return "Respiratory"
    if 520 <= v < 580 or f == 787: return "Digestive"
    if f == 250: return "Diabetes"
    if 800 <= v < 1000: return "Injury"
    if 710 <= v < 740: return "Musculoskeletal"
    if 580 <= v < 630 or f == 788: return "Genitourinary"
    if 140 <= v < 240: return "Neoplasms"
    return "Other"


def admission_type(i):
    return "Emergency" if i in (1, 2, 7) else "Elective" if i == 3 else "Other"


def discharge_group(i):
    if i in (1, 6, 8, 9, 13): return "Home"
    if i in (2, 3, 4, 5, 14, 22, 23, 24): return "Facility"
    return "Other"


def load_kaggle(path):
    """Kaggle 'Diabetes 130-US hospitals' (diabetic_data.csv). Target: readmitted within 30 days."""
    df = pd.read_csv(path, na_values="?", low_memory=False)
    df = df[df["gender"].isin(["Male", "Female"])]
    df = df[df["discharge_disposition_id"] != 11]      # patient expired - cannot be readmitted
    df = df.dropna(subset=["diag_1"])
    df = df.drop_duplicates(subset=["patient_nbr"], keep="first").copy()   # first encounter only
    df["age"] = df["age"].map(_age_to_num)
    for c in ("A1Cresult", "max_glu_serum"):
        df[c] = df[c].fillna("None").astype(str)
    df["service_utilization"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    drugs = [d for d in DRUGS if d in df.columns]
    df["numchange"] = df[drugs].isin(["Up", "Down"]).sum(axis=1)
    df["admission_type"] = df["admission_type_id"].map(admission_type)
    df["discharge_group"] = df["discharge_disposition_id"].map(discharge_group)
    df["diag_group"] = df["diag_1"].map(diag_group)
    df["target"] = (df["readmitted"] == "<30").astype(int)
    return df[FEATURES + ["target"]]


def make_synthetic(n=8000, seed=42):
    """Demo data with the same columns, so the project runs before the Kaggle CSV is downloaded."""
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": r.normal(65, 15, n).clip(18, 95).round(),
        "time_in_hospital": r.integers(1, 15, n),
        "num_lab_procedures": r.integers(1, 90, n),
        "num_procedures": r.integers(0, 6, n),
        "num_medications": r.integers(1, 35, n),
        "number_outpatient": r.poisson(0.4, n),
        "number_emergency": r.poisson(0.3, n),
        "number_inpatient": r.poisson(0.6, n),
        "number_diagnoses": r.integers(1, 10, n),
        "numchange": r.choice([0, 1, 2, 3], n, p=[.7, .22, .06, .02]),
        "gender": r.choice(["Male", "Female"], n),
        "A1Cresult": r.choice(["None", "Norm", ">7", ">8"], n, p=[.7, .1, .1, .1]),
        "max_glu_serum": r.choice(["None", "Norm", ">200", ">300"], n, p=[.9, .05, .03, .02]),
        "change": r.choice(["Ch", "No"], n, p=[.45, .55]),
        "diabetesMed": r.choice(["Yes", "No"], n, p=[.75, .25]),
        "admission_type": r.choice(["Emergency", "Elective", "Other"], n, p=[.7, .2, .1]),
        "discharge_group": r.choice(["Home", "Facility", "Other"], n, p=[.7, .2, .1]),
        "diag_group": r.choice(DIAG_GROUPS, n),
    })
    df["service_utilization"] = df.number_outpatient + df.number_emergency + df.number_inpatient
    z = (-2.3 + 0.015 * (df.age - 65) + 0.07 * (df.time_in_hospital - 4)
         + 0.015 * (df.num_medications - 16) + 0.40 * df.number_inpatient
         + 0.25 * df.number_emergency + 0.08 * df.number_outpatient
         + 0.10 * (df.number_diagnoses - 6) + 0.30 * (df.A1Cresult == ">8")
         + 0.15 * (df.diabetesMed == "Yes") + 0.10 * df.numchange
         + 0.25 * (df.discharge_group == "Facility") + 0.15 * (df.admission_type == "Emergency")
         + 0.20 * (df.diag_group == "Circulatory"))
    p = 1 / (1 + np.exp(-z))
    df["target"] = (r.random(n) < p).astype(int)
    return df
