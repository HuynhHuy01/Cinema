# Generated by Django 5.0.6 on 2024-10-24 09:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('movie_app', '0037_bookings_uuid'),
    ]

    operations = [
        migrations.AddField(
            model_name='shows',
            name='screen',
            field=models.CharField(default='Screen 1', max_length=300),
        ),
    ]
