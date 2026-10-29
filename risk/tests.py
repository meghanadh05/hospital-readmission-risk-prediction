import json
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from .models import Patient, Prediction


VALID_PAYLOAD = {
    "age": 67, "gender": "Female", "time_in_hospital": 5,
    "num_lab_procedures": 42, "num_procedures": 1, "num_medications": 14,
    "number_outpatient": 1, "number_emergency": 0, "number_inpatient": 2,
    "number_diagnoses": 6, "A1Cresult": ">7", "max_glu_serum": "None",
    "change": "Ch", "numchange": 1, "diabetesMed": "Yes",
    "admission_type": "Emergency", "discharge_group": "Home", "diag_group": "Diabetes",
}


class PageTests(TestCase):
    def test_dashboard_requires_login(self):
        response = self.client.get(reverse("dashboard"))
        self.assertRedirects(response, f"{reverse('login')}?next=/")

    def test_authenticated_user_can_open_dashboard(self):
        user = get_user_model().objects.create_user("doctor", password="safe-test-password")
        self.client.force_login(user)
        self.assertEqual(self.client.get(reverse("dashboard")).status_code, 200)

    def test_admission_creates_prediction_and_pdf(self):
        user = get_user_model().objects.create_user("doctor", password="safe-test-password")
        patient = Patient.objects.create(mrn="TEST-001", name="Test Patient", age=67, gender="Female")
        self.client.force_login(user)
        admission = {key: value for key, value in VALID_PAYLOAD.items() if key not in {"age", "gender"}}
        admission.update({"admitted_on": "2026-10-29", "notes": "Test admission"})

        response = self.client.post(reverse("admission_create", args=[patient.pk]), admission)
        prediction = Prediction.objects.get(admission__patient=patient)
        self.assertRedirects(response, reverse("prediction_detail", args=[prediction.pk]))
        self.assertGreaterEqual(prediction.probability, 0)
        self.assertLessEqual(prediction.probability, 1)

        report = self.client.get(reverse("report_pdf", args=[prediction.pk]))
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report["Content-Type"], "application/pdf")
        self.assertTrue(report.content.startswith(b"%PDF"))


class HealthTests(TestCase):
    def test_health_checks_database_and_model(self):
        response = self.client.get(reverse("health"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")


class PredictionAPITests(TestCase):
    def post(self, payload=VALID_PAYLOAD, key="test-api-key"):
        return self.client.post(
            reverse("api_predict"), data=json.dumps(payload), content_type="application/json",
            headers={"X-API-Key": key},
        )

    @override_settings(READMISSION_API_KEY="")
    def test_api_must_be_configured(self):
        self.assertEqual(self.post().status_code, 503)

    @override_settings(READMISSION_API_KEY="test-api-key")
    def test_api_rejects_invalid_key(self):
        self.assertEqual(self.post(key="wrong").status_code, 401)

    @override_settings(READMISSION_API_KEY="test-api-key")
    def test_api_returns_validation_errors(self):
        response = self.post(VALID_PAYLOAD | {"age": 200})
        self.assertEqual(response.status_code, 400)
        self.assertIn("age", response.json()["details"])

    @override_settings(READMISSION_API_KEY="test-api-key")
    @patch("risk.api.predictor.predict")
    def test_api_returns_prediction(self, predict):
        predict.return_value = (0.1761234, "High", "Candidate Model", [{"feature": "Age", "impact": 1.2}])
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["risk_level"], "High")
        self.assertIn("inference_ms", response.json())
        self.assertIn("prediction;dur=", response["Server-Timing"])
        self.assertEqual(predict.call_args.args[0]["service_utilization"], 3)


class ModelIntegrationTests(SimpleTestCase):
    def test_bundled_model_runs_real_inference(self):
        from risk.api import RiskPredictionForm
        from risk.predictor import predict

        form = RiskPredictionForm(VALID_PAYLOAD)
        self.assertTrue(form.is_valid(), form.errors)
        probability, level, model_name, factors = predict(form.predictor_values())
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)
        self.assertIn(level, {"Low", "Medium", "High"})
        self.assertTrue(model_name)
        self.assertIsInstance(factors, list)
