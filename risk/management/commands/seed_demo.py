import random
from datetime import timedelta

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.utils import timezone

from risk import predictor
from risk.models import Admission, Patient, Prediction

NAMES = ["Anita Rao", "Ravi Kumar", "Sita Devi", "Mohan Reddy", "Lakshmi Naidu", "Imran Khan",
         "Priya Sharma", "Venkat Rao", "Fatima Begum", "Suresh Babu", "Kavitha Reddy", "Arjun Das",
         "Meena Kumari", "Ramesh Goud", "Sunitha Rani", "Harish Varma", "Padma Latha", "Ali Hussain",
         "Divya Teja", "Narayana Murthy", "Swathi Pillai", "Kiran Kumar", "Bhavani Shankar", "Zoya Sheikh"]
DIAG = ["Diabetes", "Circulatory", "Respiratory", "Genitourinary", "Digestive"]


class Command(BaseCommand):
    help = "Create a demo doctor login and sample patients with predictions."

    def handle(self, *a, **k):
        if not User.objects.filter(username="doctor").exists():
            User.objects.create_superuser("doctor", "doctor@example.com", "doctor123")
        random.seed(7)
        for i, name in enumerate(NAMES, 1):
            pt, new = Patient.objects.get_or_create(
                mrn=f"MRN{1000 + i}",
                defaults={"name": name, "age": random.randint(35, 88),
                          "gender": random.choice(["Male", "Female"])})
            if not new:
                continue
            adm = Admission.objects.create(
                patient=pt, diag_group=random.choice(DIAG),
                admission_type=random.choice(["Emergency", "Emergency", "Elective"]),
                discharge_group=random.choice(["Home", "Home", "Facility"]),
                max_glu_serum=random.choice(["None", "None", "Norm", ">200"]),
                change=random.choice(["Ch", "No"]), numchange=random.choice([0, 0, 1, 2]),
                time_in_hospital=random.randint(1, 13), num_lab_procedures=random.randint(10, 80),
                num_procedures=random.randint(0, 5), num_medications=random.randint(3, 30),
                number_outpatient=random.randint(0, 3), number_emergency=random.randint(0, 3),
                number_inpatient=random.choice([0, 0, 1, 2, 3, 4]), number_diagnoses=random.randint(2, 9),
                A1Cresult=random.choice(["None", "Norm", ">7", ">8"]),
                diabetesMed=random.choice(["Yes", "No"]))
            pred = predictor.run_for_admission(adm)
            Prediction.objects.filter(pk=pred.pk).update(
                created=timezone.now() - timedelta(days=random.randint(0, 13)))
        self.stdout.write(self.style.SUCCESS("Demo data ready. Login: doctor / doctor123"))
