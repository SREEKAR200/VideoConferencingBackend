import whisper
import tempfile
import os

# Load model once when the module is imported (choose 'small' or any variant)
model = whisper.load_model("small")

def transcribe_audio_file(audio_bytes, file_extension='wav'):
    # Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp:
        tmp.write(audio_bytes)
        tmp_filepath = tmp.name

    # Run ASR
    result = model.transcribe(tmp_filepath)
    text = result.get("text", "")

    # Cleanup temp file
    os.remove(tmp_filepath)
    
    return text
