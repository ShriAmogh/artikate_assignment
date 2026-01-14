from django.urls import path
from .views import index, ask
from .views import ingestion_status  


urlpatterns = [
    path("", index),
    path("ask/", ask),
    path("status/<int:job_id>/", ingestion_status),
]


