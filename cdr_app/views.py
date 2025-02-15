from django.shortcuts import render, redirect
from .models import CallDetailRecord
from .forms import CallDetailRecordForm, CSVUploadForm, IndividualCallRecordForm
from .utils import analyze_sentiment, transcribe_audio, filter_suspect_calls, flag_suspect_calls_with_ai, summarize_text
from django.core.files.storage import FileSystemStorage
from django.db.models import Q
import os
import pandas as pd
import plotly.express as px
from transformers import pipeline  # Import pipeline from transformers
import librosa
import torch
from pydub import AudioSegment

def transcribe_audio(file_path):
    try:
        print(f"Transcribing file: {file_path}")
        
        # Convert MP3 to WAV if necessary
        if file_path.endswith('.mp3'):
            wav_file_path = file_path.rsplit('.', 1)[0] + '.wav'
            audio = AudioSegment.from_mp3(file_path)
            audio.export(wav_file_path, format="wav")
            file_path = wav_file_path
        
        transcriber = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h-lv60-self")
        audio, sample_rate = librosa.load(file_path, sr=16000)
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        result = transcriber(audio_tensor)
        print(f"Transcription result: {result['text']}")
        return result['text']
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return "No transcript available."

def analyze_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return 'N/A', 0.0
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)[0]
    return result['label'], result['score']

def filter_suspect_calls(cdrs, suspect_keywords):
    suspect_calls = []
    for cdr in cdrs:
        if any(keyword.lower() in cdr.call_notes.lower() for keyword in suspect_keywords if cdr.call_notes):
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

def summarize_text(text):
    if not isinstance(text, str) or not text.strip():
        return "No transcription available."
    summarization_pipeline = pipeline("summarization")
    summary = summarization_pipeline(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def cdr_list(request):
    query = request.GET.get('q', '')
    suspect_keywords = request.GET.getlist('suspect_keywords', [])
    use_ai_filter = request.GET.get('use_ai_filter', False) == 'true'
    
    form = CallDetailRecordForm()
    csv_form = CSVUploadForm()
    individual_form = IndividualCallRecordForm()

    if request.method == 'POST':
        if 'csv_file' in request.FILES:
            csv_form = CSVUploadForm(request.POST, request.FILES)
            if csv_form.is_valid():
                csv_file = request.FILES['csv_file']
                fs = FileSystemStorage()
                filename = fs.save(csv_file.name, csv_file)
                file_path = os.path.join(fs.location, filename)
                df = pd.read_csv(file_path)
                for index, row in df.iterrows():
                    call_id = row.get('call_id', '')
                    caller_number = row.get('caller_number', '')
                    callee_number = row.get('callee_number', '')
                    call_start_time = pd.to_datetime(row.get('call_start_time', ''))
                    call_end_time = pd.to_datetime(row.get('call_end_time', ''))
                    call_type = row.get('call_type', '')
                    call_notes = row.get('call_notes', '')

                    cdr, created = CallDetailRecord.objects.update_or_create(
                        call_id=call_id,
                        defaults={
                            'caller_number': caller_number,
                            'callee_number': callee_number,
                            'call_start_time': call_start_time,
                            'call_end_time': call_end_time,
                            'call_type': call_type,
                            'call_notes': call_notes,
                        }
                    )
                    if not cdr.call_notes:
                        if cdr.call_recording:
                            fs = FileSystemStorage()
                            file_path = os.path.join(fs.location, cdr.call_recording.name)
                            transcription = transcribe_audio(file_path)
                            cdr.call_notes = transcription
                            cdr.sentiment_label, cdr.sentiment_score = analyze_sentiment(transcription)
                            cdr.summary = summarize_text(transcription)
                            cdr.is_suspect = flag_suspect_calls_with_ai([cdr])[0].is_suspect if cdr.call_notes else False
                    else:
                        cdr.sentiment_label, cdr.sentiment_score = analyze_sentiment(cdr.call_notes)
                        cdr.summary = summarize_text(cdr.call_notes)
                        cdr.is_suspect = flag_suspect_calls_with_ai([cdr])[0].is_suspect if cdr.call_notes else False
                    cdr.save()
                return redirect('cdr_list')
        elif 'individual_call_recording' in request.POST:
            print("Individual call recording form submitted")
            individual_form = IndividualCallRecordForm(request.POST, request.FILES)
            if individual_form.is_valid():
                print("Form is valid")
                cdr = individual_form.save(commit=False)
                if 'call_recording' in request.FILES:
                    print("Call recording file is present")
                    cdr.call_recording = request.FILES['call_recording']
                    fs = FileSystemStorage()
                    filename = fs.save(cdr.call_recording.name, cdr.call_recording)
                    uploaded_file_url = fs.url(filename)
                    file_path = os.path.join(fs.location, filename)
                    transcription = transcribe_audio(file_path)
                    print(f"Transcription: {transcription}")
                    cdr.call_notes = transcription
                    cdr.sentiment_label, cdr.sentiment_score = analyze_sentiment(transcription)
                    cdr.summary = summarize_text(transcription)
                    cdr.is_suspect = flag_suspect_calls_with_ai([cdr])[0].is_suspect if cdr.call_notes else False
                else:
                    print("No call recording file uploaded")
                cdr.save()
                return redirect('cdr_list')
            else:
                print("Form is invalid")
                print(individual_form.errors)
    else:
        form = CallDetailRecordForm()
        csv_form = CSVUploadForm()
        individual_form = IndividualCallRecordForm()

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
        if not cdr.call_notes:
            if cdr.call_recording:
                fs = FileSystemStorage()
                file_path = os.path.join(fs.location, cdr.call_recording.name)
                transcription = transcribe_audio(file_path)
                cdr.call_notes = transcription
                cdr.sentiment_label, cdr.sentiment_score = analyze_sentiment(transcription)
                cdr.summary = summarize_text(transcription)
                cdr.is_suspect = flag_suspect_calls_with_ai([cdr])[0].is_suspect if cdr.call_notes else False
                cdr.save()
        if not cdr.sentiment_label:
            if cdr.call_notes:
                cdr.sentiment_label, cdr.sentiment_score = analyze_sentiment(cdr.call_notes)
            else:
                cdr.sentiment_label = 'N/A'
                cdr.sentiment_score = 0.0
            cdr.save()
        if not cdr.summary:
            if cdr.call_notes:
                cdr.summary = summarize_text(cdr.call_notes)
            else:
                cdr.summary = "No transcription available."
            cdr.save()

    return render(request, 'cdr_list.html', {
        'cdrs': cdrs,
        'form': form,
        'csv_form': csv_form,
        'individual_form': individual_form,
        'query': query,
        'suspect_keywords': suspect_keywords,
        'use_ai_filter': use_ai_filter
    })

def cdr_visualization(request):
    cdrs = CallDetailRecord.objects.all()
    df = pd.DataFrame(list(cdrs.values()))

    # Example: Call Frequency Over Time
    df['call_start_time'] = pd.to_datetime(df['call_start_time'])
    df['date'] = df['call_start_time'].dt.date
    call_frequency = df.groupby('date').size().reset_index(name='counts')
    call_start_times = call_frequency['date'].astype(str).tolist()
    call_counts = call_frequency['counts'].tolist()

    # Example: Call Type Distribution
    call_type_distribution = df['call_type'].value_counts().reset_index()
    call_type_distribution.columns = ['call_type', 'counts']
    call_types = call_type_distribution['call_type'].tolist()
    call_type_counts = call_type_distribution['counts'].tolist()

    # Pass data to the template
    context = {
        'call_start_times': call_start_times,
        'call_counts': call_counts,
        'call_types': call_types,
        'call_type_counts': call_type_counts,
        'incoming_count': df[df['call_type'] == 'incoming']['call_type'].count(),
        'outgoing_count': df[df['call_type'] == 'outgoing']['call_type'].count(),
        'missed_count': df[df['call_type'] == 'missed']['call_type'].count(),
        'voicemail_count': df[df['call_type'] == 'voicemail']['call_type'].count(),
    }

    return render(request, 'cdr_visualization.html', context)