from django.db import models

class CallDetailRecord(models.Model):
    call_id = models.CharField(max_length=100, unique=True)
    caller_number = models.CharField(max_length=20)
    callee_number = models.CharField(max_length=20)
    call_start_time = models.DateTimeField()
    call_end_time = models.DateTimeField()
    call_type = models.CharField(max_length=20, choices=[
        ('incoming', 'Incoming'),
        ('outgoing', 'Outgoing'),
        ('missed', 'Missed'),
        ('voicemail', 'Voicemail'),
    ])
    call_notes = models.TextField(blank=True, null=True)  # Optional call notes
    call_recording = models.FileField(upload_to='recordings/', blank=True, null=True)  # Store call recordings
    sentiment_label = models.CharField(max_length=20, blank=True, null=True)  # Sentiment label from sentiment analysis
    sentiment_score = models.FloatField(blank=True, null=True)  # Sentiment score from sentiment analysis
    is_suspect = models.BooleanField(default=False)  # Flag to mark suspect calls
    summary = models.TextField(blank=True, null=True)  # Summary of the call

    def __str__(self):
        return f"Call ID: {self.call_id}"

    @property
    def call_duration(self):
        if self.call_start_time and self.call_end_time:
            return int((self.call_end_time - self.call_start_time).total_seconds())
        return 0