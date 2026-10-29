from django.db import models
from django.utils import timezone


class Patient(models.Model):
    GENDERS = [("Male", "Male"), ("Female", "Female")]
    mrn = models.CharField("Medical record no.", max_length=20, unique=True)
    name = models.CharField(max_length=100)
    age = models.PositiveSmallIntegerField()
    gender = models.CharField(max_length=10, choices=GENDERS)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.mrn})"


class Admission(models.Model):
    """One hospital stay: diagnosis, treatment history, lab result and prior-visit counts."""
    A1C = [("None", "Not measured"), ("Norm", "Normal"), (">7", "> 7"), (">8", "> 8")]
    YN = [("Yes", "Yes"), ("No", "No")]
    ADM = [(x, x) for x in ("Emergency", "Elective", "Other")]
    DISCH = [("Home", "Home / home health"), ("Facility", "Transferred to another facility"), ("Other", "Other")]
    DIAG = [(x, x) for x in ("Circulatory", "Respiratory", "Digestive", "Diabetes", "Injury",
                             "Musculoskeletal", "Genitourinary", "Neoplasms", "Other")]
    GLU = [("None", "Not measured"), ("Norm", "Normal"), (">200", "> 200"), (">300", "> 300")]
    CHG = [("Ch", "Diabetes medication changed"), ("No", "No change")]
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name="admissions")
    admitted_on = models.DateField(default=timezone.localdate)
    diag_group = models.CharField("Primary diagnosis group", max_length=20, choices=DIAG, default="Diabetes")
    admission_type = models.CharField(max_length=10, choices=ADM, default="Emergency")
    discharge_group = models.CharField("Discharged to", max_length=10, choices=DISCH, default="Home")
    time_in_hospital = models.PositiveSmallIntegerField("Days in hospital", default=3)
    num_lab_procedures = models.PositiveSmallIntegerField("Lab tests done", default=40)
    num_procedures = models.PositiveSmallIntegerField("Procedures", default=1)
    num_medications = models.PositiveSmallIntegerField("Medications", default=12)
    number_outpatient = models.PositiveSmallIntegerField("Outpatient visits (past year)", default=0)
    number_emergency = models.PositiveSmallIntegerField("Emergency visits (past year)", default=0)
    number_inpatient = models.PositiveSmallIntegerField("Previous admissions (past year)", default=0)
    number_diagnoses = models.PositiveSmallIntegerField("Number of diagnoses", default=5)
    A1Cresult = models.CharField("HbA1c lab result", max_length=5, choices=A1C, default="None")
    max_glu_serum = models.CharField("Glucose serum test", max_length=5, choices=GLU, default="None")
    change = models.CharField("Change in diabetes medication", max_length=2, choices=CHG, default="No")
    numchange = models.PositiveSmallIntegerField("Number of drugs with dose changed", default=0)
    diabetesMed = models.CharField("On diabetes medication", max_length=3, choices=YN, default="Yes")
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-admitted_on", "-id"]

    def __str__(self):
        return f"{self.patient.name} - {self.admitted_on}"


class Prediction(models.Model):
    admission = models.OneToOneField(Admission, on_delete=models.CASCADE, related_name="prediction")
    probability = models.FloatField()
    risk_level = models.CharField(max_length=6)
    model_name = models.CharField(max_length=40)
    factors = models.JSONField(default=list)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-probability"]
