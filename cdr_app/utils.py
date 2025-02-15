import os
import speech_recognition as sr
from transformers import pipeline
import pandas as pd
import librosa
import torch

def transcribe_audio(file_path):
    # Load the transcription pipeline
    transcriber = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h-lv60-self")
    
    # Load the audio file
    audio, sample_rate = librosa.load(file_path, sr=16000)
    
    # Convert to tensor
    audio_tensor = torch.tensor(audio).unsqueeze(0)
    
    # Transcribe the audio
    result = transcriber(audio_tensor)
    return result['text']
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
        if cdr.call_notes and isinstance(cdr.call_notes, str):
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