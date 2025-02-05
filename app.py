import streamlit as st
import whisper
import os
import time
import torch
from audiorecorder import audiorecorder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from deepgram import DeepgramClient, SpeakOptions
import psutil

# Configuration
WHISPER_MODEL = "small"
MAX_MEMORY_PERCENT = 85  # Safety threshold

def memory_safety_check():
    """Prevent memory overload crashes"""
    mem = psutil.virtual_memory()
    if mem.percent > MAX_MEMORY_PERCENT:
        st.error(f"🚨 Memory limit exceeded ({mem.percent}%). Please refresh the app.")
        st.stop()

def save_audio(audio):
    """Save recorded audio with memory check"""
    memory_safety_check()
    try:
        filename = f"temp_{int(time.time())}.wav"
        audio.export(filename, format="wav")
        return filename
    except Exception as e:
        st.error(f"❌ Audio save failed: {str(e)}")
        return None

@st.cache_resource(max_entries=1)
def load_models():
    try:
        # First import core components
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        st.write("🔄 Loading Whisper model...")
        whisper_model = whisper.load_model(WHISPER_MODEL, device="cpu")
        
        st.write("🔄 Loading translation model...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/m2m100_418M")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/m2m100_418M",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        return whisper_model, model, tokenizer
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        raise

def transcribe_audio(model, audio_path):
    """Memory-safe transcription"""
    memory_safety_check()
    try:
        return model.transcribe(audio_path)["text"]
    except Exception as e:
        st.error(f"❌ Transcription failed: {str(e)}")
        return ""

def translate_text(text, tokenizer, model):
    """Optimized translation with memory checks"""
    memory_safety_check()
    try:
        if not text.strip():
            return ""

        tokenizer.src_lang = "ur"
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="longest"
        )
        
        generated_tokens = model.generate(
            **inputs.to(model.device),
            forced_bos_token_id=tokenizer.lang_code_to_id["en"],
            max_length=512,
            num_beams=3,
            early_stopping=True
        )
        
        return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"❌ Translation failed: {str(e)}")
        return ""

def text_to_speech(text, api_key):
    """TTS with cleanup"""
    try:
        deepgram = DeepgramClient(api_key)
        options = SpeakOptions(model="aura-asteria-en", encoding="linear16", container="wav")
        output_file = f"output_{int(time.time())}.wav"

        deepgram.speak.rest.v("1").save(output_file, {"text": text}, options)
        return output_file if os.path.exists(output_file) else None
    except Exception as e:
        st.error(f"❌ TTS failed: {str(e)}")
        return None

def cleanup_files(*filenames):
    """Guaranteed cleanup"""
    for f in filenames:
        try:
            if f and os.path.exists(f):
                os.remove(f)
        except Exception as ex:
            pass  # Silent cleanup

def main():
        # Version verification
    import transformers
    if transformers.__version__ != "4.40.1":
        st.error(f"❌ Wrong transformers version: {transformers.__version__} (Required: 4.40.1)")
        st.stop()
    st.title("🎙️ Smart Audio Translator")
    st.markdown("""
    <style>
    .stAudio { border-radius: 15px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
    .stButton>button { width: 100%; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

    # Initialize system
    whisper_model, trans_model, trans_tokenizer = load_models()
    deepgram_key = st.text_input("🔑 Deepgram API Key:", type="password")

    # Audio interface
    audio = audiorecorder("⏺️ Start Recording", "⏹️ Stop Recording", "⏸️ Pause")
    
    if audio.duration_seconds > 0:
        with st.spinner("🧠 Processing..."):
            audio_path = save_audio(audio)
            if not audio_path:
                return

            original_text = transcribe_audio(whisper_model, audio_path)
            cleanup_files(audio_path)
            
            if not original_text:
                return

            translated_text = translate_text(original_text, trans_tokenizer, trans_model)
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🗣️ Original Transcription")
                st.code(original_text, language="ur")
            with col2:
                st.subheader("🌐 English Translation")
                st.code(translated_text)

            # TTS output
            if deepgram_key and translated_text:
                audio_output = text_to_speech(translated_text, deepgram_key)
                if audio_output:
                    st.audio(audio_output)
                    cleanup_files(audio_output)

if __name__ == "__main__":
    main()