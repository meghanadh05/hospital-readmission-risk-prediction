from functools import lru_cache

import joblib
import pandas as pd
from django.conf import settings

from ml.common import FEATURES
from .models import Prediction

LABELS = {
    "age": "Age", "gender": "Gender", "time_in_hospital": "Days in hospital",
    "num_lab_procedures": "Lab tests done", "num_procedures": "Procedures",
    "num_medications": "Number of medications", "number_outpatient": "Outpatient visits",
    "number_emergency": "Emergency visits", "number_inpatient": "Previous admissions",
    "number_diagnoses": "Number of diagnoses", "A1Cresult": "HbA1c lab result",
    "diabetesMed": "On diabetes medication", "service_utilization": "Prior hospital visits (total)",
    "numchange": "Drug dose changes", "max_glu_serum": "Glucose serum test", "change": "Medication changed",
    "admission_type": "Admission type", "discharge_group": "Discharge destination",
    "diag_group": "Primary diagnosis group",
}


@lru_cache(maxsize=1)
def _bundle():
    return joblib.load(settings.MODEL_PATH)


def model_ready():
    """Return whether the serialized model can be loaded and has the expected contract."""
    try:
        bundle = _bundle()
        return all(key in bundle for key in ("pipeline", "baseline", "name", "thresholds"))
    except Exception:
        return False


def risk_level(p):
    low, high = _bundle().get("thresholds") or settings.RISK_THRESHOLDS
    return "Low" if p < low else "Medium" if p < high else "High"


def predict(values):
    """Return (probability, level, model name, top contributing factors).

    Factor impact = drop in probability (percentage points) when that single feature is
    replaced with the typical (median/mode) training value. Simple, dependency-free explainability.
    """
    b = _bundle()
    pipe = b["pipeline"]
    original = {feature: values[feature] for feature in FEATURES}
    rows = [original]
    for f in FEATURES:
        alternative = original.copy()
        alternative[f] = b["baseline"][f]
        rows.append(alternative)

    # One batched model call avoids 20 separate preprocessing and inference passes.
    scores = pipe.predict_proba(pd.DataFrame(rows, columns=FEATURES))[:, 1]
    p = float(scores[0])
    impacts = [(feature, p - float(scores[index + 1])) for index, feature in enumerate(FEATURES)]
    impacts.sort(key=lambda x: -x[1])
    factors = [{"feature": LABELS[f], "impact": round(d * 100, 1)} for f, d in impacts if d > 0.005][:5]
    return p, risk_level(p), b["name"], factors


def run_for_admission(adm):
    pt = adm.patient
    values = {"age": pt.age, "gender": pt.gender}
    values.update({f: getattr(adm, f) for f in FEATURES
                   if f not in values and f != "service_utilization"})
    values["service_utilization"] = adm.number_outpatient + adm.number_emergency + adm.number_inpatient
    p, level, name, factors = predict(values)
    obj, _ = Prediction.objects.update_or_create(
        admission=adm,
        defaults={"probability": p, "risk_level": level, "model_name": name, "factors": factors})
    return obj
