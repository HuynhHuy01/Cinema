# Generated by Django 5.0.6 on 2024-11-01 05:31

from django.conf import settings
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('movie_app', '0052_rating'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AlterUniqueTogether(
            name='rating',
            unique_together={('film', 'user')},
        ),
    ]
