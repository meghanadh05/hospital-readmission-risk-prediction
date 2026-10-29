import os

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Create or update the deployment administrator from environment variables."

    def handle(self, *args, **options):
        username = os.getenv("DJANGO_SUPERUSER_USERNAME", "admin").strip() or "admin"
        email = os.getenv("DJANGO_SUPERUSER_EMAIL", "").strip()
        password = os.getenv("DJANGO_SUPERUSER_PASSWORD", "")

        if not email or not password:
            self.stdout.write(self.style.WARNING(
                "Administrator not changed: set DJANGO_SUPERUSER_EMAIL and "
                "DJANGO_SUPERUSER_PASSWORD to enable deployment login."
            ))
            return

        user, created = get_user_model().objects.get_or_create(username=username)
        user.email = email
        user.is_staff = True
        user.is_superuser = True
        user.is_active = True
        user.set_password(password)
        user.save()
        action = "created" if created else "updated"
        self.stdout.write(self.style.SUCCESS(f"Administrator '{username}' {action}."))
