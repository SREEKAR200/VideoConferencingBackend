from ..diarization.pyannote_diarization import diarize_audio_file
from ..asr.whisper_asr import transcribe_audio_file

def process_diarize_asr_translate(audio_bytes, file_extension="wav"):
    segments = diarize_audio_file(audio_bytes, file_extension=file_extension)
    transcript = transcribe_audio_file(audio_bytes, file_extension=file_extension)

    # For now, attach the full transcript to all segments
    for seg in segments:
        seg["transcript"] = transcript

    return segments
