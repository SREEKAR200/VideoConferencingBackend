import stable_whisper
import tempfile
import os

# Load model once
model = stable_whisper.load_model("small")

def transcribe_with_timestamps(audio_bytes, file_extension='wav'):
    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp:
        tmp.write(audio_bytes)
        tmp_filepath = tmp.name

    # Run transcription with timestamps
    result = model.transcribe(tmp_filepath)

    # Extract output with timestamps
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"]
        })

    os.remove(tmp_filepath)
    return segments
