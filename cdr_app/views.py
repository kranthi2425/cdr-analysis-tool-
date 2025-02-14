from django.shortcuts import render
from .models import CallDetailRecord

def cdr_list(request):
    cdrs = CallDetailRecord.objects.all()
    return render(request, 'cdr_list.html', {'cdrs': cdrs})