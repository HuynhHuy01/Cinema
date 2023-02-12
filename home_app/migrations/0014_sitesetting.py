# Generated by Django 4.1.2 on 2023-02-11 14:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home_app', '0013_rename_sitesetting_homepageslider'),
    ]

    operations = [
        migrations.CreateModel(
            name='SiteSetting',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('site_name', models.CharField(max_length=75)),
                ('site_logo', models.FileField(upload_to='uploads/site_logo')),
                ('favicon', models.FileField(upload_to='uploads/favicon')),
                ('telegram', models.URLField()),
                ('email', models.EmailField(max_length=254)),
                ('is_active', models.BooleanField()),
            ],
        ),
    ]