import streamlit as st
import whisper
import sounddevice as sd
import wave
import pyaudio
import numpy as np
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# Remove old import from deepgram and update below in the function

# Configuration
AUDIO_DURATION = 10  # seconds
SAMPLE_RATE = 16000
WHISPER_MODEL = "small"  # Using a smaller Whisper model for reduced resource usage

def save_audio(audio, sample_rate):
    """Save recorded audio to a temporary file with a unique filename."""
    try:
        filename = f"temp_{int(time.time())}.wav"
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
        return filename
    except Exception as e:
        st.error(f"❌ Audio save failed: {str(e)}")
        return None

def load_models():
    """Load the required models without caching to reduce memory load."""
    # Use GPU if available; otherwise fallback to CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        st.write("🔄 Loading Whisper model...")
        whisper_model = whisper.load_model(WHISPER_MODEL, device=device)
        
        st.write("🔄 Loading translation model...")
        # Use a smaller translation model for lower resource consumption
        model_name = "facebook/m2m100_418M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        
        return whisper_model, model, tokenizer, device
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        raise

def record_audio():
    """Record audio using the default microphone."""
    try:
        st.write(f"🎙️ Recording for {AUDIO_DURATION} seconds...")
        audio = sd.rec(int(AUDIO_DURATION * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1,
                       dtype=np.int16)
        sd.wait()
        return audio
    except Exception as e:
        st.error(f"❌ Recording failed: {str(e)}")
        return None

def transcribe_audio(model, audio_path):
    """Transcribe the saved audio file using the Whisper model."""
    try:
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"❌ Transcription failed: {str(e)}")
        return ""

def translate_text(text, tokenizer, model):
    """Translate text using the generic translation implementation."""
    try:
        if not text.strip():
            return ""
            
        # Set source language (here assumed to be Urdu for demonstration)
        tokenizer.src_lang = "ur"
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["en"],
            max_length=1024,
            num_beams=5,
            early_stopping=True
        )
        
        return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"❌ Translation failed: {str(e)}")
        return ""

def text_to_speech(text, api_key):
    """Convert text to speech using the Deepgram SDK version 3."""
    try:
        # Import the new classes from the Deepgram SDK version 3
        from deepgram import DeepgramClient, SpeakOptions

        # Instantiate a Deepgram client with the API key
        deepgram = DeepgramClient(api_key)
        options = SpeakOptions(model="aura-asteria-en")
        output_file = f"output_{int(time.time())}.mp3"
        SPEAK_TEXT = {"text": text}

        # Call the REST interface to save the TTS output.
        # The 'v("1")' part specifies the API version.
        response = deepgram.speak.rest.v("1").save(output_file, SPEAK_TEXT, options)

        if os.path.exists(output_file):
            return output_file
        return None
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
    st.title("Audio Translator")
    st.markdown("Record audio → Get translation with voice output")
    
    try:
        whisper_model, trans_model, trans_tokenizer, device = load_models()
    except:
        st.stop()
    
    deepgram_key = st.text_input("Enter Deepgram API Key:", type="password")
    
    if st.button(f"Start Recording ({AUDIO_DURATION}s)"):
        with st.spinner("Recording..."):
            audio = record_audio()
            
        if audio is not None:
            with st.spinner("Processing..."):
                audio_path = save_audio(audio, SAMPLE_RATE)
                if audio_path:
                    original_text = transcribe_audio(whisper_model, audio_path)
                    if original_text:
                        st.subheader("Original Transcription")
                        ## st.write(original_text)
                        
                        translated_text = translate_text(original_text, trans_tokenizer, trans_model)
                        if translated_text:
                            st.subheader("Translation")
                            st.write(translated_text)
                            
                            if deepgram_key:
                                audio_output = text_to_speech(translated_text, deepgram_key)
                                if audio_output:
                                    st.audio(audio_output)
                                    cleanup_files(audio_output)
                    
                    cleanup_files(audio_path)

if __name__ == "__main__":
    main()
