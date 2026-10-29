"""Build the same 30-day-readmission feature table from MIMIC-III v1.4 CSV files.

MIMIC-III (https://physionet.org/content/mimiciii/1.4/) needs PhysioNet credentialed access.
Put the downloaded CSVs in one folder, then:   python ml/train.py path/to/mimic_folder
Required: ADMISSIONS, PATIENTS, DIAGNOSES_ICD.  Optional (used if present): PROCEDURES_ICD, PRESCRIPTIONS.
Diabetes-specific fields MIMIC lacks (A1C, glucose test, drug changes) are set to 'None' / 0.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ml.common import FEATURES, diag_group


def _read(folder, name, **kw):
    for suffix in (".csv", ".csv.gz"):
        f = Path(folder) / (name + suffix)
        if f.exists():
            df = pd.read_csv(f, low_memory=False, **kw)
            df.columns = df.columns.str.lower()
            return df
    return None


def _icd9(code):
    """MIMIC stores ICD-9 without the dot ('25000'); turn into '250.00' so ml.common.diag_group works."""
    c = str(code)
    if c[:1] in ("V", "E"):
        return c
    return c[:3] + "." + c[3:] if len(c) > 3 else c


def load_mimic(folder):
    adm = _read(folder, "ADMISSIONS")
    pat = _read(folder, "PATIENTS", usecols=lambda c: c.lower() in ("subject_id", "gender", "dob"))
    dx = _read(folder, "DIAGNOSES_ICD")
    if adm is None or pat is None or dx is None:
        raise FileNotFoundError("Need ADMISSIONS, PATIENTS and DIAGNOSES_ICD CSV files in " + str(folder))

    adm = adm[(adm.admission_type != "NEWBORN") & (adm.hospital_expire_flag == 0)].copy()
    adm["admittime"] = pd.to_datetime(adm.admittime)
    adm["dischtime"] = pd.to_datetime(adm.dischtime)
    adm = adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

    # label: next admission begins within 30 days of this discharge
    nxt = adm.groupby("subject_id").admittime.shift(-1)
    gap = (nxt - adm.dischtime).dt.days
    adm["target"] = ((gap >= 0) & (gap <= 30)).astype(int)

    # prior-year utilisation (admissions / emergency admissions in the 365 days before admission)
    prior, prior_em = np.zeros(len(adm), int), np.zeros(len(adm), int)
    for _, g in adm.groupby("subject_id"):
        t = g.admittime.values
        em = (g.admission_type == "EMERGENCY").values
        for k, i in enumerate(g.index):
            m = (t[:k] >= t[k] - np.timedelta64(365, "D"))
            prior[i], prior_em[i] = m.sum(), em[:k][m].sum()
    adm["number_inpatient"], adm["number_emergency"] = prior, prior_em
    adm["number_outpatient"] = 0

    adm = adm.merge(pat.rename(columns=str.lower), on="subject_id", how="left")
    adm["age"] = (adm.admittime.dt.year - adm.dob.astype(str).str[:4].astype(int)).clip(18, 90)  # >89 are date-shifted
    adm["gender"] = adm.gender.map({"M": "Male", "F": "Female"})
    adm["time_in_hospital"] = ((adm.dischtime - adm.admittime).dt.total_seconds() / 86400).round().clip(1, 60)

    dx = dx.dropna(subset=["icd9_code"])
    first = dx[dx.seq_num == 1].set_index("hadm_id").icd9_code.map(lambda c: diag_group(_icd9(c)))
    adm["diag_group"] = adm.hadm_id.map(first).fillna("Other")
    adm["number_diagnoses"] = adm.hadm_id.map(dx.groupby("hadm_id").size()).fillna(0)

    proc, rx = _read(folder, "PROCEDURES_ICD"), _read(folder, "PRESCRIPTIONS", usecols=lambda c: c.lower() in ("hadm_id", "drug"))
    adm["num_procedures"] = adm.hadm_id.map(proc.groupby("hadm_id").size()).fillna(0) if proc is not None else 0
    adm["num_medications"] = adm.hadm_id.map(rx.groupby("hadm_id").drug.nunique()).fillna(0) if rx is not None else 0
    adm["num_lab_procedures"] = 0   # would need LABEVENTS (very large); left out
    adm["service_utilization"] = adm.number_outpatient + adm.number_emergency + adm.number_inpatient
    adm["numchange"] = 0
    adm["A1Cresult"], adm["max_glu_serum"], adm["change"], adm["diabetesMed"] = "None", "None", "No", "No"
    adm["admission_type"] = adm.admission_type.map({"EMERGENCY": "Emergency", "URGENT": "Emergency", "ELECTIVE": "Elective"}).fillna("Other")
    loc = adm.discharge_location.fillna("").str.upper()
    adm["discharge_group"] = np.where(loc.str.startswith("HOME"), "Home",
                              np.where(loc.str.contains("SNF|REHAB|LONG TERM|SKILLED|HOSPICE|TRANSFER"), "Facility", "Other"))
    return adm[FEATURES + ["target"]].dropna(subset=["age", "gender"])
