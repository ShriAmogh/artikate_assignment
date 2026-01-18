from django.contrib.auth.models import User
from django.db import models

class UserIndex(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    collection_name = models.CharField(max_length=255, unique=True)
    #chroma_dir = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} -->  {self.collection_name}"
    
