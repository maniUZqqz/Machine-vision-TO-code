from django.contrib import admin
from .models import ConversionJob


@admin.register(ConversionJob)
class ConversionJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'status', 'processing_time_ms', 'created_at')
    list_filter = ('status',)
    readonly_fields = ('id', 'created_at', 'updated_at')
