from django.shortcuts import render, redirect
from .models import CallDetailRecord
from .forms import CallDetailRecordForm, CSVUploadForm
from .utils import analyze_sentiment, transcribe_audio, filter_suspect_calls, flag_suspect_calls_with_ai
from django.core.files.storage import FileSystemStorage
from django.db.models import Q
import os
import speech_recognition as sr
from transformers import pipeline
import pandas as pd
import plotly.express as px
from django.shortcuts import render
from .models import CallDetailRecord

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio_data)
            return transcription
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results; {e}"

def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

def filter_suspect_calls(cdrs, suspect_keywords):
    suspect_calls = []
    for cdr in cdrs:
        if any(keyword.lower() in cdr.call_notes.lower() for keyword in suspect_keywords):
            suspect_calls.append(cdr)
    return suspect_calls

def flag_suspect_calls_with_ai(cdrs):
    classifier_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["suspect", "normal"]
    suspect_calls = []
    for cdr in cdrs:
        if cdr.call_notes:
            result = classifier_pipeline(cdr.call_notes, candidate_labels=labels)
            if result['labels'][0] == "suspect":
                cdr.is_suspect = True
                cdr.save()
                suspect_calls.append(cdr)
    return suspect_calls

def cdr_list(request):
    query = request.GET.get('q', '')
    suspect_keywords = request.GET.getlist('suspect_keywords', [])
    use_ai_filter = request.GET.get('use_ai_filter', False) == 'true'
    
    if request.method == 'POST':
        if 'csv_file' in request.FILES:
            form = CSVUploadForm(request.POST, request.FILES)
            if form.is_valid():
                csv_file = request.FILES['csv_file']
                fs = FileSystemStorage()
                filename = fs.save(csv_file.name, csv_file)
                file_path = os.path.join(fs.location, filename)
                df = pd.read_csv(file_path)
                for index, row in df.iterrows():
                    call_id = row.get('call_id', '')
                    caller_number = row.get('caller_number', '')
                    callee_number = row.get('callee_number', '')
                    call_start_time = row.get('call_start_time', '')
                    call_end_time = row.get('call_end_time', '')
                    call_duration = row.get('call_duration', 0)
                    call_type = row.get('call_type', '')
                    call_notes = row.get('call_notes', '')

                    cdr, created = CallDetailRecord.objects.update_or_create(
                        call_id=call_id,
                        defaults={
                            'caller_number': caller_number,
                            'callee_number': callee_number,
                            'call_start_time': call_start_time,
                            'call_end_time': call_end_time,
                            'call_duration': call_duration,
                            'call_type': call_type,
                            'call_notes': call_notes,
                        }
                    )
                    if call_notes:
                        cdr.sentiment_label, cdr.sentiment_score = analyze_sentiment(call_notes)
                    else:
                        cdr.sentiment_label = 'N/A'
                        cdr.sentiment_score = 0.0
                    cdr.save()
                return redirect('cdr_list')
        else:
            form = CallDetailRecordForm(request.POST, request.FILES)
            if form.is_valid():
                cdr = form.save(commit=False)
                if cdr.call_recording:
                    fs = FileSystemStorage()
                    filename = fs.save(cdr.call_recording.name, cdr.call_recording)
                    uploaded_file_url = fs.url(filename)
                    file_path = os.path.join(fs.location, filename)
                    transcription = transcribe_audio(file_path)
                    cdr.call_notes = transcription
                    cdr.sentiment_label, cdr.sentiment_score = analyze_sentiment(transcription)
                cdr.save()
                return redirect('cdr_list')
    else:
        form = CallDetailRecordForm()
        csv_form = CSVUploadForm()

    cdrs = CallDetailRecord.objects.all()
    if query:
        cdrs = cdrs.filter(
            Q(call_id__icontains=query) |
            Q(caller_number__icontains=query) |
            Q(callee_number__icontains=query) |
            Q(call_notes__icontains=query)
        )

    if suspect_keywords:
        cdrs = filter_suspect_calls(cdrs, suspect_keywords)

    if use_ai_filter:
        cdrs = flag_suspect_calls_with_ai(cdrs)

    for cdr in cdrs:
        if not cdr.sentiment_label:
            if cdr.call_notes and isinstance(cdr.call_notes, str):
                cdr.sentiment_label, cdr.sentiment_score = analyze_sentiment(cdr.call_notes)
            else:
                cdr.sentiment_label = 'N/A'
                cdr.sentiment_score = 0.0
            cdr.save()

    return render(request, 'cdr_list.html', {
        'cdrs': cdrs,
        'form': form,
        'csv_form': csv_form,
        'query': query,
        'suspect_keywords': suspect_keywords,
        'use_ai_filter': use_ai_filter
    })

def cdr_visualization(request):
    cdrs = CallDetailRecord.objects.all()
    df = pd.DataFrame(list(cdrs.values()))

    # Example: Call Frequency Over Time
    fig = px.histogram(df, x='call_start_time', nbins=30, title='Call Frequency Over Time')
    graph_div = fig.to_html(full_html=False)

    return render(request, 'cdr_visualization.html', {'graph_div': graph_div})

