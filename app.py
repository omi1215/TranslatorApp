import streamlit as st
import whisper
import os
import time
import torch
from audiorecorder import audiorecorder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from deepgram import DeepgramClient, SpeakOptions

# Configuration
WHISPER_MODEL = "small"  # Whisper model size

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
        
        st.write("🔄 Loading translation model...")
        model_name = "facebook/m2m100_418M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        
        return whisper_model, model, tokenizer, device
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

def translate_text(text, tokenizer, model):
    """Translate text from Urdu to English."""
    try:
        if not text.strip():
            return ""
            
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
    """Convert text to speech using Deepgram."""
    try:
        deepgram = DeepgramClient(api_key)
        options = SpeakOptions(model="aura-asteria-en")
        output_file = f"output_{int(time.time())}.mp3"

        response = deepgram.speak.rest.v("1").save(output_file, {"text": text}, options)
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
    st.markdown("Record speech → Translate to English → Hear the translation")
    
    try:
        whisper_model, trans_model, trans_tokenizer, device = load_models()
    except:
        st.stop()
    
    deepgram_key = st.text_input("Enter Deepgram API Key:", type="password")
    
    # Audio recorder component
    audio = audiorecorder("⏺️ Start Recording", "⏹️ Stop Recording")
    
    if audio.duration_seconds > 0:
        with st.spinner("Processing..."):
            # Save and process audio
            audio_path = save_audio(audio)
            if not audio_path:
                return

            # Transcription
            original_text = transcribe_audio(whisper_model, audio_path)
            if not original_text:
                cleanup_files(audio_path)
                return

            # Translation
            translated_text = translate_text(original_text, trans_tokenizer, trans_model)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Transcription")
                st.write(original_text)
            with col2:
                st.subheader("English Translation")
                st.write(translated_text)

            # Text-to-speech
            if deepgram_key and translated_text:
                audio_output = text_to_speech(translated_text, deepgram_key)
                if audio_output:
                    st.audio(audio_output)
                    cleanup_files(audio_output)
            
            cleanup_files(audio_path)

if __name__ == "__main__":
    main()