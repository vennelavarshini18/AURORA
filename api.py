from fastapi import FastAPI
from pydantic import BaseModel
from utils import load_translation_bundle, translate_greedy

app = FastAPI(title="Aurora API", version="1.0")

#loading once at startup but not every req
hi_model, hi_src_tok, hi_tgt_tok = load_translation_bundle(
    "models/besteng2hindi.keras",
    "models/en_tokenizer.pkl",
    "models/hi_tokenizer.pkl"
)

hing_model, hing_src_tok, hing_tgt_tok = load_translation_bundle(
    "models/besteng2hinglish.keras",
    "models/eng_tokenizer.pkl",
    "models/hing_tokenizer.pkl"
)

class TranslateRequest(BaseModel):
    text: str
    target_lang: str  

@app.get("/")
def root():
    return {"message": "Aurora API is running 🚀"}

@app.post("/translate")
def translate(req: TranslateRequest):
    if not req.text.strip():
        return {"error": "Empty text provided"}

    if req.target_lang.lower() == "hindi":
        out = translate_greedy(hi_model, hi_src_tok, hi_tgt_tok, req.text)
    else:
        out = translate_greedy(hing_model, hing_src_tok, hing_tgt_tok, req.text)

    return {"input": req.text, "target_lang": req.target_lang, "translation": out}
