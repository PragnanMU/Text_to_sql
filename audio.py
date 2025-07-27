import speech_recognition as sr
import os
import tempfile
from pydub import AudioSegment
import io

def transcribe_audio(audio_path):
    """
    Transcribe audio file using local speech recognition
    """
    try:
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Convert audio file to WAV format if needed
        audio_file_path = convert_to_wav(audio_path)
        
        # Load the audio file
        with sr.AudioFile(audio_file_path) as source:
            # Read the audio data
            audio_data = recognizer.record(source)
            
            # Perform speech recognition
            text = recognizer.recognize_google(audio_data)
            
            return text
            
    except sr.UnknownValueError:
        raise Exception("Speech recognition could not understand the audio")
    except sr.RequestError as e:
        raise Exception(f"Could not request results from speech recognition service; {e}")
    except Exception as e:
        raise Exception(f"Error transcribing audio: {str(e)}")
    finally:
        # Clean up temporary file if created
        if audio_file_path != audio_path and os.path.exists(audio_file_path):
            try:
                os.remove(audio_file_path)
            except:
                pass

def convert_to_wav(audio_path):
    """
    Convert various audio formats to WAV format for speech recognition
    """
    try:
        # Get file extension
        file_extension = os.path.splitext(audio_path)[1].lower()
        
        # If already WAV, return as is
        if file_extension == '.wav':
            return audio_path
        
        # Convert other formats to WAV
        audio = AudioSegment.from_file(audio_path)
        
        # Create temporary WAV file
        temp_wav_path = tempfile.mktemp(suffix='.wav')
        audio.export(temp_wav_path, format='wav')
        
        return temp_wav_path
        
    except Exception as e:
        raise Exception(f"Error converting audio format: {str(e)}")

def record_audio():
    """
    Record audio from microphone and return the audio data
    """
    try:
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        
        with microphone as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Set timeout and phrase time limit for better control
            recognizer.energy_threshold = 4000
            recognizer.dynamic_energy_threshold = True
            
            # Record audio with timeout
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=30)
            
            return audio
            
    except sr.WaitTimeoutError:
        raise Exception("No speech detected within timeout period")
    except Exception as e:
        raise Exception(f"Error recording audio: {str(e)}")

def transcribe_audio_data(audio_data):
    """
    Transcribe audio data directly (for microphone recordings)
    """
    try:
        recognizer = sr.Recognizer()
        text = recognizer.recognize_google(audio_data)
        return text
        
    except sr.UnknownValueError:
        raise Exception("Speech recognition could not understand the audio")
    except sr.RequestError as e:
        raise Exception(f"Could not request results from speech recognition service; {e}")
    except Exception as e:
        raise Exception(f"Error transcribing audio: {str(e)}")