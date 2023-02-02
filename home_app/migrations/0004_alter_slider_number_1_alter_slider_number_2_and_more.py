# Generated by Django 4.1.2 on 2022-12-05 18:26

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('movie_app', '0020_date_is_active_film_is_active_and_more'),
        ('home_app', '0003_alter_slider_number_1_alter_slider_number_2_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='slider',
            name='number_1',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='one', to='movie_app.film', unique=True),
        ),
        migrations.AlterField(
            model_name='slider',
            name='number_2',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='two', to='movie_app.serie', unique=True),
        ),
        migrations.AlterField(
            model_name='slider',
            name='number_3',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='three', to='movie_app.serie', unique=True),
        ),
        migrations.AlterField(
            model_name='slider',
            name='number_4',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='four', to='movie_app.serie', unique=True),
        ),
        migrations.AlterField(
            model_name='slider',
            name='number_5',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='five', to='movie_app.film', unique=True),
        ),
    ]
