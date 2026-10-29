import json
from io import StringIO
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile

from django.contrib.auth import get_user_model
from django.core.management import call_command
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

    def test_only_superuser_can_import_patient_csv(self):
        user = get_user_model().objects.create_user("doctor", password="safe-test-password")
        self.client.force_login(user)
        upload = SimpleUploadedFile(
            "patients.csv", b"mrn,name,age,gender\nD-001,Anita Rao,68,Female\n", "text/csv",
        )
        self.assertEqual(self.client.post(reverse("patient_import"), {"file": upload}).status_code, 403)
        self.assertEqual(Patient.objects.count(), 0)

        admin = get_user_model().objects.create_superuser(
            "admin", "admin@example.com", "safe-admin-password",
        )
        self.client.force_login(admin)
        upload = SimpleUploadedFile(
            "patients.csv",
            b"mrn,name,age,gender\nD-001,Anita Rao,68,Female\nD-002,Ravi Kumar,57,Male\n",
            "text/csv",
        )
        response = self.client.post(reverse("patient_import"), {"file": upload}, follow=True)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Patient.objects.count(), 2)
        self.assertEqual(Patient.objects.get(mrn="D-001").name, "Anita Rao")

        update = SimpleUploadedFile(
            "patients.csv", b"mrn,name,age,gender\nD-001,Anita Updated,69,Female\n", "text/csv",
        )
        self.client.post(reverse("patient_import"), {"file": update})
        self.assertEqual(Patient.objects.count(), 2)
        self.assertEqual(Patient.objects.get(mrn="D-001").name, "Anita Updated")

    def test_csv_import_is_atomic_when_a_row_is_invalid(self):
        admin = get_user_model().objects.create_superuser(
            "admin", "admin@example.com", "safe-admin-password",
        )
        self.client.force_login(admin)
        upload = SimpleUploadedFile(
            "patients.csv",
            b"mrn,name,age,gender\nD-001,Valid Patient,68,Female\nD-002,Invalid Patient,999,Male\n",
            "text/csv",
        )
        response = self.client.post(reverse("patient_import"), {"file": upload})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(Patient.objects.count(), 0)
        self.assertContains(response, "Ensure this value is less than or equal to 120")


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


class EnsureAdminTests(TestCase):
    @patch.dict("os.environ", {
        "DJANGO_SUPERUSER_USERNAME": "admin",
        "DJANGO_SUPERUSER_EMAIL": "admin@example.com",
        "DJANGO_SUPERUSER_PASSWORD": "strong-test-password-123",
    })
    def test_command_creates_login_and_can_update_it(self):
        output = StringIO()
        call_command("ensure_admin", stdout=output)
        user = get_user_model().objects.get(username="admin")
        self.assertTrue(user.is_superuser)
        self.assertTrue(user.check_password("strong-test-password-123"))

        call_command("ensure_admin", stdout=output)
        self.assertEqual(get_user_model().objects.filter(username="admin").count(), 1)
