import os
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