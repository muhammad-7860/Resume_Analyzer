from django.db import models


class Resume(models.Model):
    resume_text = models.TextField()
    class_label = models.CharField(max_length=100)
# Create your models here.
