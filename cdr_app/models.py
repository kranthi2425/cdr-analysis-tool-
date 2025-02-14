from django.db import models

class CallDetailRecord(models.Model):
    call_id = models.CharField(max_length=100)
    caller_number = models.CharField(max_length=20)
    callee_number = models.CharField(max_length=20)
    call_start_time = models.DateTimeField()
    call_end_time = models.DateTimeField()
    call_duration = models.IntegerField()  # in seconds
    call_type = models.CharField(max_length=20)  # e.g., 'incoming', 'outgoing'

    def __str__(self):
        return f"Call ID: {self.call_id}"