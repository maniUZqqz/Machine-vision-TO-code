from django.core.exceptions import ImproperlyConfigured

from .base import SECRET_KEY

DEBUG = False

if SECRET_KEY.startswith("django-insecure"):
    raise ImproperlyConfigured("DJANGO_ENV=production needs a real SECRET_KEY in the environment (or .env).")
