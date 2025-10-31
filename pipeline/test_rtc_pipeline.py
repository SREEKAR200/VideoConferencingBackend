import requests

with open("C:/SpeechVideoProject/data/samples/tone-test.mp3", "rb") as f:
    audio = f.read()

response = requests.post(
    "http://localhost:8000/rtc_full_pipeline",
    files={"file": ("sample_audio.wav", audio, "audio/wav")},
    data={"src_lang": "hindi", "tgt_lang": "english"}
)

print(response.json())
