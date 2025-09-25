from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

# Create your models here.


class Present(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.localdate)
    present = models.BooleanField(default=False)


class Time(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.localdate)
    time = models.DateTimeField(null=True, blank=True)
    out = models.BooleanField(default=False)
