# Generated by Django 4.1.1 on 2022-11-06 10:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('movie_app', '0012_film_background_alter_film_banner'),
    ]

    operations = [
        migrations.AddField(
            model_name='serie',
            name='background',
            field=models.ImageField(blank=True, null=True, upload_to='uploads/backgrounds'),
        ),
    ]
