import os
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import sparse_categorical_crossentropy 

class transblock(tf.keras.layers.Layer):
    def __init__(self, embdim, heads, ffdim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embdim = embdim
        self.heads = heads
        self.ffdim = ffdim
        self.rate = rate
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=embdim)
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(ffdim, activation="relu"),
            tf.keras.layers.Dense(embdim),
        ])
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dp1 = tf.keras.layers.Dropout(rate)
        self.dp2 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        cfg = super().get_config().copy()
        cfg.update({
            "embdim": self.embdim,
            "heads": self.heads,
            "ffdim": self.ffdim,
            "rate": self.rate,
        })
        return cfg

    def call(self, x, training=None):
        att = self.att(x, x)
        att = self.dp1(att, training=training)
        out1 = self.ln1(x + att)
        ffout = self.ff(out1)
        ffout = self.dp2(ffout, training=training)
        return self.ln2(out1 + ffout)


class tokposemb(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab, embdim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab = vocab
        self.embdim = embdim
        self.tokemb = tf.keras.layers.Embedding(input_dim=vocab, output_dim=embdim)
        self.posemb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embdim)

    def get_config(self):
        cfg = super().get_config().copy()
        cfg.update({
            "maxlen": self.maxlen,
            "vocab": self.vocab,
            "embdim": self.embdim,
        })
        return cfg

    def call(self, x):
        ln = tf.shape(x)[-1]
        pos = tf.range(start=0, limit=ln, delta=1)
        pos = self.posemb(pos)
        x = self.tokemb(x)
        return x + pos

_url_pat = re.compile(r'https?://\S+|www\.\S+')
def remurl(text: str) -> str:
    if not isinstance(text, str): return ""
    return _url_pat.sub("", text)

try:
    import contractions as contracs
    def expandcontracs(text: str) -> str:
        if not isinstance(text, str): return ""
        return contracs.fix(text)
except Exception:
    def expandcontracs(text: str) -> str:
        return text if isinstance(text, str) else ""

_eng_keep_pat = re.compile(r'[^a-zA-Z0-9\s]')
_hi_keep_pat  = re.compile(r'[^\u0900-\u097F\s]')

def prep2(text: str, lang: str = "english") -> str:
    if not isinstance(text, str): return ""
    if lang == "english":
        return _eng_keep_pat.sub("", text)
    elif lang == "hindi":
        return _hi_keep_pat.sub("", text)
    return text

from textblob import Word
import nltk
try: nltk.data.find("corpora/stopwords")
except LookupError: nltk.download("stopwords")
try: nltk.data.find("corpora/wordnet")
except LookupError: nltk.download("wordnet")
try: nltk.data.find("corpora/omw-1.4")
except LookupError: nltk.download("omw-1.4")

from nltk.corpus import stopwords as nltk_stopwords
_eng_stop = set(nltk_stopwords.words("english"))

_hinstopwords = set([
    "और","के","का","की","से","है","हैं","यह","था","थे",
    "हो","में","पर","को","उनका","उनकी","उनके","उसका","उसकी",
    "इसके","इसका","इसकी","जो","कि","तो","ही","भी","लेकिन","क्योंकि",
    "जैसे","जब","तक","अगर","या"
])

def prep(text: str, lang: str = "english") -> str:
    if not isinstance(text, str): return ""
    if lang == "english":
        words = text.split()
        words = [Word(w).lemmatize() for w in words if w not in _eng_stop]
        return " ".join(words)
    elif lang == "hindi":
        words = text.split()
        words = [w for w in words if w not in _hinstopwords]
        return " ".join(words)
    return text

def preprocess_english_for_inference(text: str) -> str:
    t = remurl(text)
    t = expandcontracs(t)
    t = prep2(t, lang="english")
    t = prep(t, lang="english")
    return t.strip()

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_translation_bundle(model_path: str, src_tok_path: str, tgt_tok_path: str):
    custom_objects = {
        "transblock": transblock,
        "tokposemb": tokposemb,
    }
    model = load_model(model_path, compile=False, custom_objects=custom_objects)
    src_tok = load_pickle(src_tok_path)
    tgt_tok = load_pickle(tgt_tok_path)
    return model, src_tok, tgt_tok

_MXLEN = 150

def translate_greedy(model, src_tokenizer, tgt_tokenizer, english_text: str) -> str:
    cleaned = preprocess_english_for_inference(english_text)
    seq = src_tokenizer.texts_to_sequences([cleaned])
    padseq = pad_sequences(seq, maxlen=_MXLEN, padding="post", truncating="post")
    preds = model.predict(padseq, verbose=0)
    token_ids = preds[0].argmax(axis=-1).tolist()
    out_text = tgt_tokenizer.sequences_to_texts([token_ids])[0]
    return out_text.strip()

import google.generativeai as genai
from dotenv import load_dotenv

def init_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env")
    genai.configure(api_key=api_key)

def detect_language_variant(text: str) -> str:
    """Detect whether the text is in Hindi (Devanagari) or Hinglish (Romanized Hindi)."""
    if re.search(r"[\u0900-\u097F]", text):  # Devanagari range
        return "Hindi"
    return "Hinglish"

def query_gemini(translated_text: str, content_type: str, style: str) -> str:
    init_gemini()
    model = genai.GenerativeModel("gemini-1.5-flash")

    lang_variant = detect_language_variant(translated_text)

    prompt = f"""
You are given text written in **{lang_variant}**.

--- SOURCE ---
{translated_text}
--- END ---

Task:
- Create a {content_type.strip()}.
- Tone/style should be {style.strip()}.
- Keep it coherent and engaging.
- IMPORTANT: The output must strictly be in {lang_variant}, not in any other script or language.
"""

    resp = model.generate_content(prompt)
    return (resp.text or "").strip()
