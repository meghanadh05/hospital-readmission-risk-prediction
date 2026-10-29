from django import forms
from .models import Admission, Patient


class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ["mrn", "name", "age", "gender"]


class PatientCSVImportForm(forms.Form):
    file = forms.FileField(
        label="Patient CSV file",
        help_text="UTF-8 CSV, maximum 2 MB. Required columns: mrn, name, age, gender.",
    )

    def clean_file(self):
        upload = self.cleaned_data["file"]
        if upload.size > 2 * 1024 * 1024:
            raise forms.ValidationError("The CSV file must be 2 MB or smaller.")
        if not upload.name.lower().endswith(".csv"):
            raise forms.ValidationError("Upload a file with a .csv extension.")
        return upload


class PatientCSVRowForm(forms.Form):
    mrn = forms.CharField(max_length=20)
    name = forms.CharField(max_length=100)
    age = forms.IntegerField(min_value=0, max_value=120)
    gender = forms.ChoiceField(choices=Patient.GENDERS)


class AdmissionForm(forms.ModelForm):
    class Meta:
        model = Admission
        exclude = ["patient"]
        widgets = {"admitted_on": forms.DateInput(attrs={"type": "date"}, format="%Y-%m-%d"),
                   "notes": forms.Textarea(attrs={"rows": 2})}
