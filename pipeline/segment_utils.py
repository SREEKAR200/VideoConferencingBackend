from pydub import AudioSegment
import tempfile
import os

def standardize_audio(audio_bytes, file_extension="mp3", out_extension="wav", target_sr=16000):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_in:
        tmp_in.write(audio_bytes)
        in_path = tmp_in.name

    sound = AudioSegment.from_file(in_path)
    sound = sound.set_channels(1).set_frame_rate(target_sr)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{out_extension}") as tmp_out:
        sound.export(tmp_out.name, format=out_extension)
        out_path = tmp_out.name
    os.remove(in_path)
    return out_path
