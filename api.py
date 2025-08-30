import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from utils import load_pickle, transblock, tokposemb, translate_greedy

app = FastAPI(title="Aurora API", version="1.0")

try:
    hi_model = load_model(
        "models/besteng2hindi.keras",
        compile=False,
        custom_objects={"transblock": transblock, "tokposemb": tokposemb}
    )
    hi_src_tok = load_pickle("models/en_tokenizer.pkl")
    hi_tgt_tok = load_pickle("models/hi_tokenizer.pkl")
except Exception as e:
    raise RuntimeError(f"Failed loading Hindi model: {e}")

try:
    hing_model = load_model(
        "models/besteng2hinglish.keras",
        compile=False,
        custom_objects={"transblock": transblock, "tokposemb": tokposemb}
    )
    hing_src_tok = load_pickle("models/eng_tokenizer.pkl")
    hing_tgt_tok = load_pickle("models/hing_tokenizer.pkl")
except Exception as e:
    raise RuntimeError(f"Failed loading Hinglish model: {e}")

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

    return {
        "input": req.text,
        "target_lang": req.target_lang,
        "translation": out
    }
