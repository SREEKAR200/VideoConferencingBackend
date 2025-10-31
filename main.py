from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from .pipeline.process_diarized_asr import process_diarize_asr_translate
from .asr.whisper_asr import transcribe_audio_file
from .translation.nllb_translation import translate_text
from .diarization.pyannote_diarization import diarize_audio_file
from .alignment.stable_ts import transcribe_with_timestamps
from .transcript.transcript_agg import aggregate_transcript, format_transcript
from .utils.logger import setup_logging, log_request, log_error, log_success
from pydantic import BaseModel
import uvicorn

from dotenv import load_dotenv
load_dotenv()


# ----- FastAPI App -----
app = FastAPI(
    title="Speech Conferencing API",
    description="Diarization, ASR, and Translation for Indian languages and more. Upload audio and get diarization, transcript, and translation.",
    version="1.0"
)

setup_logging()

@app.get("/", summary="API health check")
def read_root():
    return {"msg": "Speech conferencing backend API running."}

@app.get("/status", summary="API status and available endpoints")
def health_check():
    endpoints = [
        "/asr", "/diarize", "/rtc_full_pipeline", "/translate", "/asr_timestamps", "/diarize_transcribe", "/transcript"
    ]
    return {
        "status": "running",
        "endpoints": endpoints,
        "languages": [
            "hindi", "marathi", "gujarati", "bengali", "punjabi", "tamil", "telugu", "malayalam", "kannada", "oriya",
            "assamese", "urdu", "sanskrit", "english"
        ]
    }

# --- Speech to Text ---
@app.post("/asr", summary="Simple ASR (audio-to-text)")
async def transcribe(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        text = transcribe_audio_file(audio_bytes)
        log_success("ASR complete")
        return JSONResponse(content={"transcript": text})
    except Exception as e:
        log_error(str(e))
        return JSONResponse({"error": str(e)})

# --- Translate text ---
@app.post("/translate", summary="Translate text between supported languages")
async def translate_api(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        src_lang = data.get("src_lang", "hindi")
        tgt_lang = data.get("tgt_lang", "english")
        translated = translate_text(text, src_lang=src_lang, tgt_lang=tgt_lang)
        log_success("Translation complete")
        return JSONResponse(content={"translation": translated})
    except Exception as e:
        log_error(str(e))
        return JSONResponse({"error": str(e)})

# --- Diarization: Speaker label and timestamps ---
@app.post("/diarize", summary="Get speaker segments from audio")
async def diarize_api(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        ext = file.filename.split('.')[-1]
        segments = diarize_audio_file(audio_bytes, file_extension=ext)
        log_success("Diarization complete")
        return {"segments": segments}
    except Exception as e:
        log_error(str(e))
        return {"error": str(e)}

@app.post("/diarize/bytes", summary="Diarization from raw bytes")
async def diarize_bytes_api(request: Request):
    try:
        audio_bytes = await request.body()
        segments = diarize_audio_file(audio_bytes, file_extension="wav")
        log_success("Diarization complete (bytes)")
        return {"segments": segments}
    except Exception as e:
        log_error(str(e))
        return {"error": str(e)}

# --- Diarization + ASR ---
@app.post("/diarize_transcribe", summary="Return speaker segments with transcripts")
async def diarize_transcribe_api(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        ext = file.filename.split('.')[-1]
        combined_results =process_diarize_asr_translate(audio_bytes, file_extension=ext)
        log_success("Diarize+ASR complete")
        return {"segments": combined_results}
    except Exception as e:
        log_error(str(e))
        return {"error": str(e)}

# --- ASR with timestamps ---
@app.post("/asr_timestamps", summary="Get transcript with word/segment timestamps")
async def transcribe_with_ts(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        segments = transcribe_with_timestamps(audio_bytes)
        log_success("ASR+Timestamps complete")
        return JSONResponse(content={"segments": segments})
    except Exception as e:
        log_error(str(e))
        return JSONResponse({"error": str(e)})

# --- Aggregate transcript using diarization and ASR+timestamps ---
@app.post("/transcript", summary="Get rich transcript from segments")
async def get_transcript(data: dict):
    try:
        asr_segments = data["asr_segments"]       # from ASR with timestamps
        diarization_segments = data["diarization"] # from diarization
        translations = data.get("translations")    # translated asr segments (optional)
        transcript = aggregate_transcript(asr_segments, diarization_segments, translations)
        formatted = format_transcript(transcript)
        log_success("Transcript aggregation complete")
        return JSONResponse(content={"transcript": transcript, "pretty": formatted})
    except Exception as e:
        log_error(str(e))
        return JSONResponse({"error": str(e)})

# --- Full pipeline: Diarization + ASR + Translation ---
@app.post("/rtc_full_pipeline", summary="Full pipeline: diarization, ASR, and translation")
async def rtc_full_pipeline_api(
    file: UploadFile = File(...),
    src_lang: str = "hindi",
    tgt_lang: str = "english"
):
    try:
        audio_bytes = await file.read()
        ext = file.filename.split('.')[-1]
        log_request({"filename": file.filename, "src_lang": src_lang, "tgt_lang": tgt_lang})
        results = process_diarize_asr_translate(audio_bytes, ext, src_lang, tgt_lang)
        log_success("Full pipeline complete")
        return {"segments": results}
    except Exception as e:
        log_error(str(e))
        return {"error": str(e)}
    
@app.post("/process_audio_chunk")
async def process_audio_chunk(file: UploadFile = File(...), src_lang: str = "hindi", tgt_lang: str = "english"):
    audio_bytes = await file.read()
    result = process_diarize_asr_translate(audio_bytes, "wav", src_lang, tgt_lang)
    return JSONResponse(content=result)

@app.post("/export_transcript")
async def export_transcript(data: dict):
    transcript = data.get("transcript", "")
    response = JSONResponse(content=transcript)
    response.headers["Content-Disposition"] = "attachment; filename=transcript.txt"
    return response



# --- Universal error handler (optional) ---
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log_error(str(exc))
    return JSONResponse(status_code=500, content={"error": str(exc)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
