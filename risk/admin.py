from django.contrib import admin
from .models import Admission, Patient, Prediction

admin.site.register(Patient)
admin.site.register(Admission)
admin.site.register(Prediction)
