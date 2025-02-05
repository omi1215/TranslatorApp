import streamlit as st
import whisper
import sounddevice as sd
import wave
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from deepgram import DeepgramClient, SpeakOptions
import urduhack
from fuzzywuzzy import fuzz, process
import langid
import stanza

# Configuration
AUDIO_DURATION = 10  # seconds
SAMPLE_RATE = 16000
WHISPER_MODEL = "base"

def save_audio(audio, sample_rate):
    """Save recorded audio to temporary file with unique filename"""
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

@st.cache_resource
def load_models():
    """Load and verify all required models"""
    try:
        st.write("🔄 Loading Whisper model...")
        whisper_model = whisper.load_model(WHISPER_MODEL)
        
        st.write("🔄 Loading translation model...")
        model_name = "facebook/m2m100_1.2B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        return whisper_model, model, tokenizer
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        raise

def record_audio():
    """Record audio with proper error handling and Streamlit's audio recorder"""
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
    """Improved transcription with language specification"""
    try:
        result = model.transcribe(
            audio_path,
            language="ur",  # Set to Urdu for transcription
            task="transcribe",
            fp16=False
        )
        return result["text"]
    except Exception as e:
        st.error(f"❌ Transcription failed: {str(e)}")
        return ""

def correct_urdu_spelling(text):
    """Correct spelling mistakes in Urdu text using urduhack"""
    try:
        if not text:
            return text  # Return original if empty or None
        
        # Use urduhack for spelling correction
        corrected_text = urduhack.correct(text)
        return corrected_text
    except Exception as e:
        st.error(f"❌ Spell-checking failed: {str(e)}")
        return text

def translate_text(text, tokenizer, model):
    """Urdu to English translation"""
    try:
        if not text.strip():
            return ""
            
        tokenizer.src_lang = "ur"  # Set source language to Urdu
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["en"],  # Target language is English
            max_length=1024,
            num_beams=5,
            early_stopping=True
        )
        
        return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"❌ Translation failed: {str(e)}")
        return ""

def text_to_speech(text, api_key):
    """Updated Deepgram API usage"""
    try:
        client = DeepgramClient(api_key)
        options = SpeakOptions(
            model="aura-asteria-en",
        )
        
        # Generate unique output filename
        output_file = f"output_{int(time.time())}.mp3"
        
        # Use the correct API endpoint and method
        response = client.speak.rest.v("1").save(
            output_file,
            {"text": text},
            options=options
        )
        
        # Verify the file was created
        if os.path.exists(output_file):
            return output_file
        else:
            st.error("❌ TTS failed: Output file not created")
            return None
    except Exception as e:
        st.error(f"❌ TTS failed: {str(e)}")
        return None

def cleanup_files(*filenames):
    """Clean up temporary files"""
    for f in filenames:
        if f and os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass

def remove_punctuation(word):
    """Remove specific punctuation marks"""
    exclude_chars = set("،.٫-۔")
    return ''.join(char for char in word if char not in exclude_chars)

def suggest_most_matched_words(incorrect_word, word_list):
    """Suggest words with similarity to incorrect word"""
    similarity_scores = process.extract(incorrect_word, word_list, scorer=fuzz.ratio)
    return [word for word, _ in similarity_scores]

def is_urdu(text):
    """Check if the input text is in Urdu"""
    lang, _ = langid.classify(text)
    return lang == 'ur'

def urdu_grammar_checker(sentence):
    """Check grammar for errors in the sentence"""
    if not is_urdu(sentence):
        print("Input is not in Urdu.")
        return []
    
    stanza.download('ur')
    nlp = stanza.Pipeline('urdu')

    doc = nlp(sentence)
    errors = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.upos == 'VERB' and word.feats and 'VerbForm=Fin' not in word.feats:
                errors.append(word.text)
    return errors

def main():
    st.title("🇵🇰 Urdu to English Translator 🗣️➡️📝➡️🇬🇧")
    st.markdown("Record Urdu audio → Get accurate English translation with voice output")
    
    try:
        whisper_model, trans_model, trans_tokenizer = load_models()
    except:
        st.stop()
    
    deepgram_key = st.text_input("🔑 Enter Deepgram API Key:", type="password")
    
    if st.button(f"🎤 Start Recording ({AUDIO_DURATION}s)"):

        with st.spinner("Recording..."):
            audio = record_audio()
            
        if audio is not None:
            with st.spinner("Saving audio..."):
                audio_path = save_audio(audio, SAMPLE_RATE)
                
            if audio_path:
                with st.spinner("Transcribing..."):
                    urdu_text = transcribe_audio(whisper_model, audio_path)
                
                if urdu_text:
                    st.subheader("Urdu Transcription")
                    st.success(urdu_text)
                    
                    # Correct the spelling of the Urdu text (if possible)
                    corrected_urdu_text = correct_urdu_spelling(urdu_text)
                    st.subheader("Corrected Urdu Text")
                    st.success(corrected_urdu_text)
                    
                    # Grammar checking
                    error_words = urdu_grammar_checker(corrected_urdu_text)
                    if error_words:
                        st.subheader("Grammar Issues")
                        st.write(f"Found grammar issues with words: {', '.join(error_words)}")
                    
                    with st.spinner("Translating..."):
                        english_text = translate_text(corrected_urdu_text, trans_tokenizer, trans_model)
                    
                    if english_text:
                        st.subheader("English Translation")
                        st.success(english_text)
                        
                        if deepgram_key:
                            with st.spinner("Generating speech..."):
                                audio_output = text_to_speech(english_text, deepgram_key)
                            
                            if audio_output:
                                st.audio(audio_output)
                                cleanup_files(audio_output)
                
                cleanup_files(audio_path)

if __name__ == "__main__":
    main()
