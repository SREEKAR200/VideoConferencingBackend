from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
# Find all supported BOS language code tokens
print(tokenizer.additional_special_tokens)
print(tokenizer.additional_special_tokens_ids)
