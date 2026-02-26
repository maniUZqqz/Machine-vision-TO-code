import uuid
from django.db import models


class ConversionJob(models.Model):
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PROCESSING = 'processing', 'Processing'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    original_image = models.ImageField(upload_to='uploads/%Y/%m/')
    languages = models.JSONField(
        default=list,
        help_text='List of language codes, e.g. ["fa", "en"]'
    )
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING
    )
    error_message = models.TextField(blank=True, default='')
    processing_time_ms = models.IntegerField(null=True, blank=True)
    generated_html = models.TextField(blank=True, default='')
    generated_css = models.TextField(blank=True, default='')
    combined_output = models.TextField(blank=True, default='')
    analysis_metadata = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Job {self.id} ({self.status})"
