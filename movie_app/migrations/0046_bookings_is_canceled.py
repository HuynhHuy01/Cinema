# Generated by Django 5.0.6 on 2024-10-29 04:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('movie_app', '0045_alter_shows_price'),
    ]

    operations = [
        migrations.AddField(
            model_name='bookings',
            name='is_canceled',
            field=models.BooleanField(default=False),
        ),
    ]
