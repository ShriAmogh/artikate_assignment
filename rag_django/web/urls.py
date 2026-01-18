from django.urls import path
from .views import dashboard, ask
from .views import ingestion_status  


urlpatterns = [
    path("", dashboard, name = 'dashboard' ),
    path("ask/", ask),
    path("status/<int:job_id>/", ingestion_status),
]


