# Generated by Django 5.0.6 on 2024-10-15 03:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('movie_app', '0030_shows_bookings'),
    ]

    operations = [
        migrations.RenameField(
            model_name='shows',
            old_name='time',
            new_name='end_time',
        ),
        migrations.AddField(
            model_name='shows',
            name='start_time',
            field=models.TimeField(default='09:00:00'),
        ),
    ]
