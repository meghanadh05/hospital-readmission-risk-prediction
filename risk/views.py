import io
import json
import csv
from datetime import timedelta

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.db.models import Count, Q
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from . import predictor
from .forms import AdmissionForm, PatientCSVImportForm, PatientCSVRowForm, PatientForm
from .models import Patient, Prediction

FOLLOW_UP = {
    "High": ["Phone call to the patient within 48 hours of discharge",
             "Follow-up clinic appointment within 7 days",
             "Medication reconciliation and adherence counselling",
             "Arrange home-health or community (ASHA/nurse) visit"],
    "Medium": ["Follow-up appointment within 14 days",
               "Discharge education and symptom warning signs",
               "Reminder call within one week"],
    "Low": ["Standard discharge instructions",
            "Routine follow-up as scheduled"],
}


def _metrics():
    return json.loads(settings.METRICS_PATH.read_text()) if settings.METRICS_PATH.exists() else None


@login_required
def dashboard(request):
    preds = Prediction.objects.select_related("admission__patient")
    counts = {r["risk_level"]: r["n"] for r in preds.values("risk_level").annotate(n=Count("id"))}
    today = timezone.localdate()
    days = [today - timedelta(days=i) for i in range(13, -1, -1)]
    chart = {
        "dist": [counts.get(k, 0) for k in ("Low", "Medium", "High")],
        "days": [d.strftime("%d %b") for d in days],
        "trend": [preds.filter(created__date=d, risk_level="High").count() for d in days],
    }
    return render(request, "risk/dashboard.html", {
        "total_patients": Patient.objects.count(), "total_preds": preds.count(),
        "high_count": counts.get("High", 0),
        "high_risk": preds.filter(risk_level="High")[:10], "chart": chart, "metrics": _metrics(),
    })


@login_required
def patient_list(request):
    q = request.GET.get("q", "").strip()
    patients = Patient.objects.all()
    if q:
        patients = patients.filter(Q(name__icontains=q) | Q(mrn__icontains=q))
    return render(request, "risk/patient_list.html", {"patients": patients, "q": q})


@login_required
def patient_import(request):
    if not request.user.is_superuser:
        raise PermissionDenied
    form = PatientCSVImportForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        try:
            content = form.cleaned_data["file"].read().decode("utf-8-sig")
        except UnicodeDecodeError:
            form.add_error("file", "The CSV must be UTF-8 encoded.")
        else:
            reader = csv.DictReader(io.StringIO(content))
            required = {"mrn", "name", "age", "gender"}
            headers = set(reader.fieldnames or [])
            if not required.issubset(headers):
                missing = ", ".join(sorted(required - headers))
                form.add_error("file", f"Missing required columns: {missing}.")
            else:
                rows, errors, seen = [], [], set()
                for line_number, row in enumerate(reader, start=2):
                    row_form = PatientCSVRowForm(row)
                    mrn = (row.get("mrn") or "").strip()
                    if mrn in seen:
                        errors.append(f"Row {line_number}: duplicate MRN '{mrn}' in this file.")
                    elif not row_form.is_valid():
                        detail = "; ".join(
                            f"{field}: {', '.join(messages)}"
                            for field, messages in row_form.errors.items()
                        )
                        errors.append(f"Row {line_number}: {detail}")
                    else:
                        seen.add(mrn)
                        rows.append(row_form.cleaned_data)

                if not rows and not errors:
                    errors.append("The CSV contains no patient rows.")
                if errors:
                    form.add_error("file", " ".join(errors[:10]))
                else:
                    created = updated = 0
                    with transaction.atomic():
                        for row in rows:
                            _, was_created = Patient.objects.update_or_create(
                                mrn=row["mrn"],
                                defaults={
                                    "name": row["name"],
                                    "age": row["age"],
                                    "gender": row["gender"],
                                },
                            )
                            created += int(was_created)
                            updated += int(not was_created)
                    messages.success(request, f"CSV import complete: {created} created, {updated} updated.")
                    return redirect("patient_list")
    return render(request, "risk/patient_import.html", {"form": form})


@login_required
def patient_csv_template(request):
    if not request.user.is_superuser:
        raise PermissionDenied
    response = HttpResponse(
        "mrn,name,age,gender\nDEMO-001,Anita Rao,68,Female\nDEMO-002,Ravi Kumar,57,Male\n",
        content_type="text/csv",
    )
    response["Content-Disposition"] = 'attachment; filename="patient_import_template.csv"'
    return response


@login_required
def patient_create(request):
    form = PatientForm(request.POST or None)
    if form.is_valid():
        p = form.save()
        messages.success(request, "Patient added.")
        return redirect("patient_detail", pk=p.pk)
    return render(request, "risk/form.html", {"form": form, "title": "Add patient"})


@login_required
def patient_detail(request, pk):
    patient = get_object_or_404(Patient, pk=pk)
    return render(request, "risk/patient_detail.html",
                  {"patient": patient, "admissions": patient.admissions.select_related("prediction")})


@login_required
def admission_create(request, pk):
    patient = get_object_or_404(Patient, pk=pk)
    form = AdmissionForm(request.POST or None)
    if form.is_valid():
        adm = form.save(commit=False)
        adm.patient = patient
        adm.save()
        pred = predictor.run_for_admission(adm)
        return redirect("prediction_detail", pk=pred.pk)
    return render(request, "risk/form.html",
                  {"form": form, "title": f"New admission - {patient.name}", "button": "Save & predict risk"})


@login_required
def prediction_detail(request, pk):
    pred = get_object_or_404(Prediction.objects.select_related("admission__patient"), pk=pk)
    return render(request, "risk/prediction_detail.html", {
        "pred": pred, "pct": round(pred.probability * 100, 1), "follow_up": FOLLOW_UP[pred.risk_level]})


@login_required
def report_pdf(request, pk):
    pred = get_object_or_404(Prediction.objects.select_related("admission__patient"), pk=pk)
    adm, pt = pred.admission, pred.admission.patient
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800

    def line(text, size=11, dy=18, font="Helvetica"):
        nonlocal y
        c.setFont(font, size)
        c.drawString(50, y, text)
        y -= dy

    line("Hospital Readmission Risk Report", 18, 30, "Helvetica-Bold")
    line(f"Patient: {pt.name}   MRN: {pt.mrn}   Age: {pt.age}   Gender: {pt.gender}")
    line(f"Admission date: {adm.admitted_on}   Diagnosis: {adm.diag_group or '-'}", dy=28)
    line(f"Predicted readmission probability: {pred.probability * 100:.1f}%", 14, 22, "Helvetica-Bold")
    line(f"Risk level: {pred.risk_level}   (model: {pred.model_name})", dy=28)
    line("Key contributing factors", 13, 22, "Helvetica-Bold")
    for f in pred.factors or [{"feature": "No single dominant factor", "impact": 0}]:
        line(f"  - {f['feature']}  (+{f['impact']} percentage points)")
    y -= 10
    line("Suggested follow-up care", 13, 22, "Helvetica-Bold")
    for s in FOLLOW_UP[pred.risk_level]:
        line(f"  - {s}")
    y -= 20
    line("Decision-support only. Final clinical judgement rests with the treating doctor.", 9)
    c.save()
    resp = HttpResponse(buf.getvalue(), content_type="application/pdf")
    resp["Content-Disposition"] = f'inline; filename="risk_report_{pt.mrn}.pdf"'
    return resp
