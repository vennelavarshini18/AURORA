import os
import re
import pickle
import ast
import operator as _op
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
import streamlit as st

def init_gemini():
    """
    Initialize Google Generative AI (Gemini) client.

    Behavior:
    - Prefer Streamlit secrets (st.secrets["GEMINI_API_KEY"]) when available (deployed on Streamlit Cloud).
    - Fallback to local .env variable (loaded via python-dotenv) when Streamlit secrets are not available (local dev).
    - Raises RuntimeError if no key is found.
    """
    api_key = None
    # First try Streamlit secrets (works on Streamlit Community Cloud)
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

    if not api_key:
        # Fallback for local development: load from .env
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY", "")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in Streamlit secrets or local .env")

    genai.configure(api_key=api_key)

def detect_language_variant(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text):
        return "Hindi"
    return "Hinglish"

def query_gemini(english_text: str, translated_text: str, content_type: str, style: str) -> str:
    init_gemini()
    model = genai.GenerativeModel("gemini-1.5-flash")
    lang_variant = detect_language_variant(translated_text)
    prompt = f"""
You are a content assistant. You will receive an English input and its {lang_variant} translation.

ENGLISH INPUT:
{english_text}

{lang_variant.upper()} TRANSLATION:
{translated_text}

Task:
- Produce a {content_type.strip()} in the tone/style: {style.strip()}.
- Preserve intent from the English input and the translation.
- Output must be strictly in {lang_variant}. Do NOT switch scripts:
  - If Hinglish, use Latin letters only.
  - If Hindi, use Devanagari only.

Output only the generated {content_type.strip()}.
"""
    resp = model.generate_content(prompt)
    return (getattr(resp, "text", None) or "").strip()

def polish_style(translated_text: str, style: str, max_words: int = 140) -> str:
    init_gemini()
    model = genai.GenerativeModel("gemini-1.5-flash")
    lang_variant = detect_language_variant(translated_text)
    prompt = f"""
Rewrite the following {lang_variant} text in the style: {style}.
- Fix grammar and fluency.
- Keep meaning faithful to the source.
- Limit length to ~{max_words} words.
- Output strictly in {lang_variant} only.
- If Hinglish, DO NOT use Devanagari. If Hindi, DO NOT use Latin.

SOURCE:
{translated_text}
"""
    resp = model.generate_content(prompt)
    return (getattr(resp, "text", None) or "").strip()

_AUDIENCE_GUIDE = {
    "Kids": "Use simple words, friendly tone, short sentences, add warmth.",
    "Boss / Work Email": "Be concise, polite, professional, clearly structured.",
    "Romantic Partner": "Be affectionate, warm, intimate but tasteful.",
    "Social Media Post": "Be catchy, compact, emoji-friendly if natural.",
    "Friend": "Casual, relatable, a bit playful.",
    "Parent / Elder": "Respectful, gentle, considerate and clear.",
    "Teacher": "Polite, respectful, to the point.",
    "Customer": "Empathetic, solution-oriented, clear next steps.",
}

def adapt_for_audience(text: str, audience: str) -> str:
    init_gemini()
    model = genai.GenerativeModel("gemini-1.5-flash")
    lang_variant = detect_language_variant(text)
    guidance = _AUDIENCE_GUIDE.get(audience, "Keep tone appropriate to the audience.")
    prompt = f"""
Adapt the following {lang_variant} text for this audience: {audience}.
Guidelines: {guidance}

Constraints:
- Preserve the core meaning.
- Use {lang_variant} only (no script switching).
- Keep it concise and natural.

TEXT:
{text}
"""
    resp = model.generate_content(prompt)
    return (getattr(resp, "text", None) or "").strip()

try:
    import wikipedia
except Exception:
    wikipedia = None

try:
    from duckduckgo_search import ddg
except Exception:
    ddg = None

try:
    from langchain.agents import initialize_agent, AgentType
    from langchain.tools import Tool
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    initialize_agent = None
    AgentType = None
    Tool = None
    ChatGoogleGenerativeAI = None

def _wiki_tool_func(q: str) -> str:
    if not wikipedia:
        return "Wikipedia library not installed."
    try:
        hits = wikipedia.search(q, results=5)
        if not hits:
            return "No Wikipedia results found."
        pieces = []
        for title in hits[:3]:
            try:
                summ = wikipedia.summary(title, sentences=2)
            except Exception:
                summ = ""
            pieces.append(f"{title}: {summ}")
        return "\n".join(pieces)
    except Exception as e:
        return f"Wikipedia error: {e}"

def _duck_tool_func(q: str) -> str:
    if not ddg:
        return "DuckDuckGo search library not installed."
    try:
        results = ddg(q, max_results=5)
        if not results:
            return "No web results found."
        bullets = []
        for r in results[:4]:
            title = r.get("title") or ""
            snippet = r.get("body") or r.get("snippet") or ""
            href = r.get("href") or r.get("url") or ""
            short = f"{title} — {snippet}"
            if href:
                short += f" ({href})"
            bullets.append(short)
        return "\n".join(bullets)
    except Exception as e:
        return f"DuckDuckGo error: {e}"

_allowed_ops = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Pow: _op.pow,
    ast.USub: _op.neg,
    ast.UAdd: lambda x: x,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant):  
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants allowed")
    if isinstance(node, ast.Num):  # 
        return node.n
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _allowed_ops:
            raise ValueError("Operator not allowed")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _allowed_ops[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _allowed_ops:
            raise ValueError("Unary operator not allowed")
        operand = _safe_eval(node.operand)
        return _allowed_ops[op_type](operand)
    raise ValueError("Unsupported expression")

def _math_tool_func(q: str) -> str:
    try:
        expr = q.strip()
        node = ast.parse(expr, mode='eval').body
        val = _safe_eval(node)
        return str(val)
    except Exception:
        return "Math tool supports simple arithmetic expressions (numbers and + - * / **)."

def _keywords_tool(q: str) -> str:
    if not isinstance(q, str) or not q.strip():
        return "No input text."
    words = re.findall(r"\w+", q.lower())
    words = [w for w in words if w not in _eng_stop and len(w) > 2]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    top = [w for w, _ in sorted_words[:7]]
    if not top:
        return "No strong keywords found."
    return ", ".join(top)

def get_langchain_agent():
    if initialize_agent is None or Tool is None or ChatGoogleGenerativeAI is None:
        return None
    tools = []
    try:
        tools.append(Tool.from_function(_wiki_tool_func, name="wikipedia_search",
                                        description="Use for factual summaries about topics (input: short topic or phrase). Return concise summary lines."))
    except Exception:
        pass
    try:
        tools.append(Tool.from_function(_duck_tool_func, name="web_search",
                                        description="Use for up-to-date web snippets; good for local tips, lists, small how-tos. Input: query."))
    except Exception:
        pass
    try:
        tools.append(Tool.from_function(_math_tool_func, name="math_solver",
                                        description="Use for simple arithmetic or numeric calculations. Input: expression."))
    except Exception:
        pass
    try:
        tools.append(Tool.from_function(_keywords_tool, name="keyword_extractor",
                                        description="Use to extract short list of keywords from input to help refine searches. Input: text."))
    except Exception:
        pass

    if not tools:
        return None

    try:
        init_gemini()
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
        )
        return agent
    except Exception:
        return None

def enrich_with_knowledge(original_text: str, translated_text: str, max_items: int = 3) -> str:
    agent = get_langchain_agent()
    lang_variant = detect_language_variant(translated_text)
    if agent:
        try:
            instruction = (
                "You are a grounded assistant that can call tools. "
                "Decide whether calling tools will yield helpful factual/contextual information for the user's input. "
                "If you call tools, synthesize their outputs into a short, practical result in the user's language variant.\n\n"
                f"Language variant required for final output: {lang_variant}.\n"
                "Output format if using tools:\n"
                "Domain: <one-word domain> — <one-line rationale>\n"
                "- suggestion 1\n"
                "- suggestion 2\n"
                f"- suggestion up to {max_items}\n\n"
                "If NO tool is useful, respond exactly with: NO_TOOL\n\n"
                "Inputs:\n"
                f"English: \"{original_text}\"\n"
                f"Translation: \"{translated_text}\"\n"
            )
            agent_response = agent.run(instruction)
            if isinstance(agent_response, str) and agent_response.strip():
                if agent_response.strip().upper() != "NO_TOOL":
                    return agent_response.strip()
        except Exception:
            pass

    init_gemini()
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
You are a practical, grounded assistant. You will receive:
- an English input
- its {lang_variant} translation

English input:
{original_text}

{lang_variant} translation:
{translated_text}

Task:
1) Choose exactly one domain from [Travel, Food, People/Compliments, Movies/Music, Education, Productivity, Health, Culture, Other].
2) In one sentence, state the chosen domain and a one-line rationale grounded in the English input (quote a short phrase from the English input).
3) Provide {max_items} concise, practical, culturally-relevant suggestions (actionable tips, checklist items, or short recommendations).
Formatting:
Domain: <Domain> — <rationale>
- suggestion 1
- suggestion 2
- suggestion 3

IMPORTANT:
- Output must be strictly in {lang_variant}. If Hinglish, use Latin letters only; if Hindi, use Devanagari only.
- Keep suggestions short, actionable, and human-friendly (no lecture tone).
Now produce the output.
"""
    resp = model.generate_content(prompt)
    return (getattr(resp, "text", None) or "").strip()

STYLE_PRESETS = [
    "Formal", "Casual", "Poetic", "Cinematic", "Bollywood Dialogue",
    "Motivational Speech", "News Headline", "Meme Caption",
    "Storyteller", "Romantic", "Inspirational", "Witty One-liner"
]

AUDIENCE_PRESETS = list(_AUDIENCE_GUIDE.keys())
