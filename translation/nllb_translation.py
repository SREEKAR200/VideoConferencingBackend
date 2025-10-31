def translate_text(text, src_lang="hindi", tgt_lang="english"):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

    # Use full Indian language code map
    INDIAN_LANG_CODES = {
        "hindi": "hin_Deva",
        "marathi": "mar_Deva",
        "gujarati": "guj_Gujr",
        "bengali": "ben_Beng",
        "punjabi": "pan_Guru",
        "tamil": "tam_Taml",
        "telugu": "tel_Telu",
        "malayalam": "mal_Mlym",
        "kannada": "kan_Knda",
        "oriya": "ory_Orya",
        "assamese": "asm_Beng",
        "urdu": "urd_Arab",
        "sanskrit": "san_Deva",
        "english": "eng_Latn"
    }
    src_code = INDIAN_LANG_CODES.get(src_lang.lower(), "hin_Deva")
    tgt_code = INDIAN_LANG_CODES.get(tgt_lang.lower(), "eng_Latn")
    tokenizer.src_lang = src_code
    inputs = tokenizer(text, return_tensors="pt")

    # Map language code to BOS token ID using additional_special_tokens
    def get_bos_token_id(tokenizer, lang_code):
        tokens = tokenizer.additional_special_tokens
        ids = tokenizer.additional_special_tokens_ids
        if lang_code not in tokens:
            lang_code = "eng_Latn"
        idx = tokens.index(lang_code)
        return ids[idx]

    bos_token_id = get_bos_token_id(tokenizer, tgt_code)
    output_tokens = model.generate(
        **inputs,
        forced_bos_token_id=bos_token_id
    )
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)
