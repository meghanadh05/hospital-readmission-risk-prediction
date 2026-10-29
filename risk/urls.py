from django.contrib.auth import views as auth
from django.urls import path

from . import api, views

urlpatterns = [
    path("health/", api.health, name="health"),
    path("api/v1/predict/", api.predict, name="api_predict"),
    path("", views.dashboard, name="dashboard"),
    path("login/", auth.LoginView.as_view(), name="login"),
    path("logout/", auth.LogoutView.as_view(), name="logout"),
    path("patients/", views.patient_list, name="patient_list"),
    path("patients/import/", views.patient_import, name="patient_import"),
    path("patients/import/template/", views.patient_csv_template, name="patient_csv_template"),
    path("patients/add/", views.patient_create, name="patient_create"),
    path("patients/<int:pk>/", views.patient_detail, name="patient_detail"),
    path("patients/<int:pk>/admit/", views.admission_create, name="admission_create"),
    path("predictions/<int:pk>/", views.prediction_detail, name="prediction_detail"),
    path("predictions/<int:pk>/pdf/", views.report_pdf, name="report_pdf"),
]
