def aggregate_transcript(asr_segments, diarization_segments, translations=None):
    """
    asr_segments: List[Dict] with 'start', 'end', 'text' (from Stable-TS)
    diarization_segments: List[Dict] with 'speaker', 'start', 'end'
    translations: List[str] or List[Dict] (optional), aligned to asr_segments
    Returns a structured transcript list.
    """
    transcript = []
    translation_iter = iter(translations) if translations else None

    for seg in asr_segments:
        speaker_label = "Unknown"
        # Find diarization segment overlapping (simple method)
        for d in diarization_segments:
            if d["start"] <= seg["start"] < d["end"]:  # If ASR segment overlaps diarization
                speaker_label = f"Speaker {d['speaker']}"
                break
        transcript_item = {
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker_label,
            "text": seg["text"]
        }
        if translation_iter:
            transcript_item["translation"] = next(translation_iter)
        transcript.append(transcript_item)
    return transcript

def format_transcript(transcript):
    """
    Returns a human-readable transcript string.
    """
    lines = []
    for item in transcript:
        lines.append(
            f"[{item['start']:.2f}-{item['end']:.2f}s] {item['speaker']}:\n  {item['text']}"
        )
        if "translation" in item:
            lines.append(f"    EN: {item['translation']}")
    return "\n".join(lines)
