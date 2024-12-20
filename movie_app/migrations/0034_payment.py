# Generated by Django 5.0.6 on 2024-10-22 05:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('movie_app', '0033_remove_shows_serie'),
    ]

    operations = [
        migrations.CreateModel(
            name='Payment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('order_id', models.IntegerField(blank=True, default=0, null=True)),
                ('amount', models.FloatField(blank=True, default=0.0, null=True)),
                ('order_desc', models.CharField(blank=True, max_length=200, null=True)),
                ('vnp_TransactionNo', models.CharField(blank=True, max_length=200, null=True)),
                ('vnp_ResponseCode', models.CharField(blank=True, max_length=200, null=True)),
            ],
        ),
    ]
