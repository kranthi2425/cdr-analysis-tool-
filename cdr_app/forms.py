from django import forms
from .models import CallDetailRecord

class CallDetailRecordForm(forms.ModelForm):
    class Meta:
        model = CallDetailRecord
        fields = '__all__'
        widgets = {
            'call_start_time': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
            'call_end_time': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
        }

class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(label='Upload CSV File')

class IndividualCallRecordForm(forms.ModelForm):
    class Meta:
        model = CallDetailRecord
        fields = ['call_id', 'caller_number', 'callee_number', 'call_start_time', 'call_end_time', 'call_type', 'call_recording']
        widgets = {
            'call_start_time': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
            'call_end_time': forms.DateTimeInput(attrs={'type': 'datetime-local'}),
        }