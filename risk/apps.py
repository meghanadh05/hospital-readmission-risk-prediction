from django.apps import AppConfig


class RiskConfig(AppConfig):
    name = "risk"

    def ready(self):
        from django.conf import settings

        if settings.PRELOAD_MODEL:
            from .predictor import model_ready

            if not model_ready():
                raise RuntimeError("The configured readmission model could not be loaded")
