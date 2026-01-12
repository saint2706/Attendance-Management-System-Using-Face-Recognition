# Generated migration for populating direction field from out field

from django.db import migrations


def populate_direction_from_out(apps, schema_editor):
    """Populate direction field based on the out field value."""
    Time = apps.get_model("users", "Time")

    # Update records in batches for efficiency
    # out=False -> direction='in'
    Time.objects.filter(out=False).update(direction="in")

    # out=True -> direction='out'
    Time.objects.filter(out=True).update(direction="out")


def reverse_populate_out_from_direction(apps, schema_editor):
    """Reverse migration: populate out field from direction field."""
    Time = apps.get_model("users", "Time")

    # direction='in' -> out=False
    Time.objects.filter(direction="in").update(out=False)

    # direction='out' -> out=True
    Time.objects.filter(direction="out").update(out=True)


class Migration(migrations.Migration):
    dependencies = [
        ("users", "0015_add_direction_to_time"),
    ]

    operations = [
        migrations.RunPython(
            populate_direction_from_out,
            reverse_code=reverse_populate_out_from_direction,
        ),
    ]
