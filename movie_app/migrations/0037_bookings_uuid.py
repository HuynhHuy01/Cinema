# Generated by Django 5.0.6 on 2024-10-24 09:09

import uuid
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('movie_app', '0036_remove_bookings_uuid'),
    ]

    operations = [
        migrations.AddField(
            model_name='bookings',
            name='uuid',
            field=models.UUIDField(default=uuid.uuid4, editable=False, unique=True),
        ),
    ]
