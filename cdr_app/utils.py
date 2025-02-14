import os
import speech_recognition as sr
from transformers import pipeline
import pandas as pd

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