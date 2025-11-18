"""Add database indexes for attendance lookup patterns."""

import django.db.models.deletion
import django.utils.timezone
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("users", "0011_alter_present_date_alter_present_id_and_more"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AlterField(
            model_name="present",
            name="user",
            field=models.ForeignKey(
                db_index=True,
                help_text="The user this attendance record belongs to.",
                on_delete=django.db.models.deletion.CASCADE,
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AlterField(
            model_name="present",
            name="date",
            field=models.DateField(
                db_index=True,
                default=django.utils.timezone.localdate,
                help_text="The date of the attendance record.",
            ),
        ),
        migrations.AlterField(
            model_name="time",
            name="user",
            field=models.ForeignKey(
                db_index=True,
                help_text="The user this time entry belongs to.",
                on_delete=django.db.models.deletion.CASCADE,
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AlterField(
            model_name="time",
            name="date",
            field=models.DateField(
                db_index=True,
                default=django.utils.timezone.localdate,
                help_text="The date of the time entry.",
            ),
        ),
        migrations.AlterField(
            model_name="time",
            name="time",
            field=models.DateTimeField(
                blank=True,
                db_index=True,
                help_text="The exact time of the event.",
                null=True,
            ),
        ),
        migrations.AddIndex(
            model_name="present",
            index=models.Index(
                fields=["user", "date"],
                name="users_present_user_date_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="present",
            index=models.Index(
                fields=["date", "user"],
                name="users_present_date_user_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="time",
            index=models.Index(
                fields=["user", "date"],
                name="users_time_user_date_idx",
            ),
        ),
        migrations.AddIndex(
            model_name="time",
            index=models.Index(
                fields=["date", "user"],
                name="users_time_date_user_idx",
            ),
        ),
    ]
