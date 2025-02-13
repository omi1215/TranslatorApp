import streamlit as st
import whisper
import os
import time
import torch
from audiorecorder import audiorecorder
from deepgram import DeepgramClient, SpeakOptions
from googletrans import Translator

# Configuration
WHISPER_MODEL = "small"  # Using a small model for faster performance

def save_audio(audio):
    """Save recorded audio to a temporary file."""
    try:
        filename = f"temp_{int(time.time())}.wav"
        audio.export(filename, format="wav")
        return filename
    except Exception as e:
        st.error(f"❌ Audio save failed: {str(e)}")
        return None

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        st.write("🔄 Loading Whisper model...")
        whisper_model = whisper.load_model(WHISPER_MODEL, device=device)
        
        st.write("🔄 Initializing translator...")
        translator = Translator()  # Auto-detection will be used for source language
        return whisper_model, translator, device
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        raise

def transcribe_audio(model, audio_path):
    """Transcribe audio using Whisper."""
    try:
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"❌ Transcription failed: {str(e)}")
        return ""

def translate_text_google(text, translator):
    """Translate text to English using googletrans with auto-detection."""
    try:
        if not text.strip():
            return ""
        translation = translator.translate(text, dest='en')  # Auto-detect source language
        return translation.text
    except Exception as e:
        st.error(f"❌ Translation failed: {str(e)}")
        return ""

def text_to_speech(text, api_key):
    """Convert text to speech using Deepgram's TTS REST API."""
    try:
        deepgram = DeepgramClient(api_key)
        options = SpeakOptions(model="aura-asteria-en")
        output_file = f"output_{int(time.time())}.mp3"
        SPEAK_OPTIONS = {"text": text}
        response = deepgram.speak.rest.v("1").save(output_file, SPEAK_OPTIONS, options)
        return output_file if os.path.exists(output_file) else None
    except Exception as e:
        st.error(f"❌ TTS failed: {str(e)}")
        return None

def cleanup_files(*filenames):
    """Remove temporary files."""
    for f in filenames:
        if f and os.path.exists(f):
            try:
                os.remove(f)
            except Exception as ex:
                st.error(f"Error cleaning up file {f}: {ex}")

def main():
    st.title("🎙️ Audio Translator")
    st.markdown("Record your voice (in Urdu or Hindi) → Get an English transcription → Hear the synthesized English speech")
    
    try:
        whisper_model, translator, device = load_models()
    except:
        st.stop()
    
    deepgram_key = st.text_input("Enter Deepgram API Key:", type="password")
    
    # Audio recorder component
    audio = audiorecorder("⏺️ Start Recording", "⏹️ Stop Recording")
    
    if audio.duration_seconds > 0:
        with st.spinner("Processing..."):
            audio_path = save_audio(audio)
            if not audio_path:
                return

            original_text = transcribe_audio(whisper_model, audio_path)
            if not original_text:
                cleanup_files(audio_path)
                return

            translated_text = translate_text_google(original_text, translator)

            if deepgram_key and translated_text:
                audio_output = text_to_speech(translated_text, deepgram_key)
                if audio_output:
                    st.audio(audio_output)
                    cleanup_files(audio_output)
            
            cleanup_files(audio_path)

if __name__ == "__main__":
    main()
