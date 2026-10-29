import json
import secrets
import time

from django import forms
from django.conf import settings
from django.db import connection
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from . import predictor
from .models import Admission, Patient


class RiskPredictionForm(forms.Form):
    age = forms.IntegerField(min_value=0, max_value=120)
    gender = forms.ChoiceField(choices=Patient.GENDERS)
    time_in_hospital = forms.IntegerField(min_value=1, max_value=365)
    num_lab_procedures = forms.IntegerField(min_value=0)
    num_procedures = forms.IntegerField(min_value=0)
    num_medications = forms.IntegerField(min_value=0)
    number_outpatient = forms.IntegerField(min_value=0)
    number_emergency = forms.IntegerField(min_value=0)
    number_inpatient = forms.IntegerField(min_value=0)
    number_diagnoses = forms.IntegerField(min_value=0)
    A1Cresult = forms.ChoiceField(choices=Admission.A1C)
    max_glu_serum = forms.ChoiceField(choices=Admission.GLU)
    change = forms.ChoiceField(choices=Admission.CHG)
    numchange = forms.IntegerField(min_value=0)
    diabetesMed = forms.ChoiceField(choices=Admission.YN)
    admission_type = forms.ChoiceField(choices=Admission.ADM)
    discharge_group = forms.ChoiceField(choices=Admission.DISCH)
    diag_group = forms.ChoiceField(choices=Admission.DIAG)

    def predictor_values(self):
        values = self.cleaned_data.copy()
        values["service_utilization"] = (
            values["number_outpatient"] + values["number_emergency"] + values["number_inpatient"]
        )
        return values


def _error(message, status, details=None):
    body = {"error": message}
    if details is not None:
        body["details"] = details
    return JsonResponse(body, status=status)


def _authorized(request):
    configured = settings.READMISSION_API_KEY
    if not configured:
        return None
    supplied = request.headers.get("X-API-Key", "")
    if not supplied:
        authorization = request.headers.get("Authorization", "")
        if authorization.startswith("Bearer "):
            supplied = authorization[7:]
    return secrets.compare_digest(supplied, configured)


@require_GET
def health(request):
    checks = {"database": False, "model": predictor.model_ready()}
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            checks["database"] = cursor.fetchone()[0] == 1
    except Exception:
        pass
    healthy = all(checks.values())
    return JsonResponse(
        {"status": "ok" if healthy else "unavailable", "checks": checks},
        status=200 if healthy else 503,
    )


@csrf_exempt
@require_POST
def predict(request):
    authorized = _authorized(request)
    if authorized is None:
        return _error("Prediction API is not configured", 503)
    if not authorized:
        return _error("Invalid or missing API key", 401)
    if request.content_type != "application/json":
        return _error("Content-Type must be application/json", 415)
    try:
        payload = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _error("Request body must contain valid JSON", 400)
    if not isinstance(payload, dict):
        return _error("JSON body must be an object", 400)

    form = RiskPredictionForm(payload)
    if not form.is_valid():
        return _error("Validation failed", 400, form.errors.get_json_data())

    started = time.perf_counter()
    probability, level, model_name, factors = predictor.predict(form.predictor_values())
    inference_ms = round((time.perf_counter() - started) * 1000, 2)
    response = JsonResponse({
        "probability": round(probability, 6),
        "percentage": round(probability * 100, 1),
        "risk_level": level,
        "model": model_name,
        "factors": factors,
        "inference_ms": inference_ms,
        "disclaimer": "Decision support only; final clinical judgement rests with the treating doctor.",
    })
    response["Server-Timing"] = f"prediction;dur={inference_ms}"
    return response
