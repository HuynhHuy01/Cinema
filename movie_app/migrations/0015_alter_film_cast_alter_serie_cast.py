# Generated by Django 4.1.1 on 2022-11-06 11:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('movie_app', '0014_film_cast_serie_cast'),
    ]

    operations = [
        migrations.AlterField(
            model_name='film',
            name='cast',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
        migrations.AlterField(
            model_name='serie',
            name='cast',
            field=models.CharField(blank=True, max_length=200, null=True),
        ),
    ]
