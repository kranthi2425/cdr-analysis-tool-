from django import forms
from django import forms
from .models import CallDetailRecord

class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(label='Upload CSV File')

class CallDetailRecordForm(forms.ModelForm):
    class Meta:
        model = CallDetailRecord
        fields = '__all__'