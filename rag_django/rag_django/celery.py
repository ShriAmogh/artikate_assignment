import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "rag_django.settings")

app = Celery("rag_django")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

app.conf.worker_pool = "solo" 
