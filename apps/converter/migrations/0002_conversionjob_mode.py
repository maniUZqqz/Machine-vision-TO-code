from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("converter", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="conversionjob",
            name="mode",
            field=models.CharField(
                default="responsive",
                help_text="responsive (flex/table) or exact (absolute)",
                max_length=16,
            ),
        ),
    ]
