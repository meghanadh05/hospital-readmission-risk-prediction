from django import forms
from .models import Admission, Patient


class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ["mrn", "name", "age", "gender"]


class AdmissionForm(forms.ModelForm):
    class Meta:
        model = Admission
        exclude = ["patient"]
        widgets = {"admitted_on": forms.DateInput(attrs={"type": "date"}, format="%Y-%m-%d"),
                   "notes": forms.Textarea(attrs={"rows": 2})}
