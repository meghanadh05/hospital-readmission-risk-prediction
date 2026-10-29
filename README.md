# Hospital Readmission Risk Prediction System

A Django application and calibrated ML model that estimate 30-day readmission risk. Doctors can manage patient admissions, review Low/Medium/High risk results and contributing factors, monitor a dashboard, and download PDF reports. A stateless JSON prediction API is included for integrations.

The bundled model was trained on the included Diabetes 130-US Hospitals dataset. Logistic Regression, Random Forest and XGBoost are compared, and the best held-out model is calibrated before deployment. This is decision support, not a substitute for clinical judgement.

## Local setup

Python 3.12 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py seed_demo
python manage.py runserver
```

Open <http://127.0.0.1:8000>. The demo command creates `doctor` / `doctor123`; change or remove this account before deployment.

On the deployed Patients page, a logged-in superuser can select **Import CSV** to create or update patients in bulk. Patient-only files need `mrn,name,age,gender`. The downloadable extended template also creates admissions and runs predictions, which immediately populates the dashboard. The whole file is validated before the import is committed.

To enable the JSON API locally, export values based on `.env.example` and set a strong `READMISSION_API_KEY`. Django does not automatically load `.env` files.

## Prediction API

`POST /api/v1/predict/` is stateless: it returns a prediction but does not save the request or patient data. Send the API key in `X-API-Key` (or as a Bearer token) and JSON containing all model inputs.

The response includes `inference_ms` and a `Server-Timing` header. Production sets `PRELOAD_MODEL=True`, so Gunicorn loads the model before accepting traffic. The patient score plus all explanation counterfactuals are evaluated in one batch to avoid request timeouts.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict/ \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: replace-with-your-key' \
  -d '{
    "age": 67, "gender": "Female", "time_in_hospital": 5,
    "num_lab_procedures": 42, "num_procedures": 1, "num_medications": 14,
    "number_outpatient": 1, "number_emergency": 0, "number_inpatient": 2,
    "number_diagnoses": 6, "A1Cresult": ">7", "max_glu_serum": "None",
    "change": "Ch", "numchange": 1, "diabetesMed": "Yes",
    "admission_type": "Emergency", "discharge_group": "Home",
    "diag_group": "Diabetes"
  }'
```

`GET /health/` checks database connectivity and confirms that the model artifact is present. It is suitable for a deployment health check.

## Deploy

The repository includes a Render Blueprint in `render.yaml`. Keep that file at the root of the Git repository, then [create a Blueprint in Render](https://render.com/docs/infrastructure-as-code). It explicitly uses Render's free web and PostgreSQL plans, installs dependencies, collects static files, runs migrations, and generates the production secret and API key. Render prompts for the administrator email and password. Every deployment safely creates or updates that administrator, so adding or changing the password and redeploying restores access.

Render's free web service sleeps after 15 minutes without traffic and may take about one minute to wake. Its free PostgreSQL database expires after 30 days, so this configuration is intended for a project demonstration rather than permanent clinical use.

For another platform, set these environment variables and run the equivalent commands:

- Required: `SECRET_KEY`, `DEBUG=False`, `ALLOWED_HOSTS`, `DATABASE_URL`
- API: `READMISSION_API_KEY`
- HTTPS site: `CSRF_TRUSTED_ORIGINS=https://your-domain.example`
- Optional hardening after HTTPS works: `SECURE_SSL_REDIRECT=True`, `SECURE_HSTS_SECONDS=31536000`

Build command:

```bash
./build.sh
```

Start command:

```bash
gunicorn config.wsgi:application --bind 0.0.0.0:${PORT:-8000}
```

On platforms other than Render, create a real administrator instead of deploying demo credentials:

```bash
python manage.py createsuperuser
```

## Data and training

`ml/data/diabetic_data.csv` is included, so `python ml/train.py` compares Logistic Regression, Random Forest and XGBoost, calibrates the best held-out model, and replaces the artifact. The bundled artifact in `ml/artifacts/` is already trained. Other datasets and the optional MIMIC-III loader are described in `ml/data/DATASETS.md`.

## Validation

```bash
python manage.py check
python manage.py check --deploy
python manage.py test
```

`check --deploy` intentionally reports HTTPS warnings unless the corresponding production environment variables are enabled. Django's optional HSTS-preload warning remains because preload should only be enabled manually after the final domain and HTTPS setup are stable.
