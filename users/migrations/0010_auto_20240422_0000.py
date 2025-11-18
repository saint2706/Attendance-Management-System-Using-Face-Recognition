from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):
    dependencies = [
        ("users", "0009_auto_20190703_1816"),
    ]

    operations = [
        migrations.AlterField(
            model_name="present",
            name="date",
            field=models.DateField(default=django.utils.timezone.localdate),
        ),
        migrations.AlterField(
            model_name="time",
            name="date",
            field=models.DateField(default=django.utils.timezone.localdate),
        ),
    ]
