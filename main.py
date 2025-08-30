import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import requests
from utils import (
    load_translation_bundle,
    translate_greedy,
    query_gemini,
    polish_style,
    adapt_for_audience,
    enrich_with_knowledge,
    STYLE_PRESETS,
    AUDIENCE_PRESETS,
)

API_BASE = os.environ.get("AURORA_API_BASE", "https://aurora-api.onrender.com")
API_URL = f"{API_BASE.rstrip('/')}/translate"

st.set_page_config(page_title="AURORA", layout="wide")
st.markdown(
    """
    <style>
    :root {
      --bg: #f6f7fb;
      --card: rgba(255,255,255,0.75);
      --muted: #6b7280;
      --text: #0f172a;
      --accent1: #7c4dff;
      --accent2: #3dd1d6;
      --radius: 20px;
      --input-bg: rgba(255,255,255,0.85);
      --input-border: rgba(15,23,42,0.08);
      --shadow: 0 10px 28px rgba(16,24,40,0.1);
      --card-border: rgba(15,23,42,0.06);
      --sidebar-bg: linear-gradient(180deg, #fff, #f9f9ff);
      --button-shadow: 0 10px 26px rgba(124,77,255,0.25);
      --placeholder: rgba(15,23,42,0.35);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #050a1a;
        --card: rgba(15,18,30,0.8);
        --muted: #9ca3af;
        --text: #e6eef6;
        --accent1: #8b5cf6;
        --accent2: #06b6d4;
        --radius: 20px;
        --input-bg: rgba(12,14,24,0.85);
        --input-border: rgba(255,255,255,0.05);
        --shadow: 0 12px 34px rgba(2,6,23,0.55);
        --card-border: rgba(255,255,255,0.05);
        --sidebar-bg: linear-gradient(180deg, #071025, #0b1128);
        --button-shadow: 0 10px 28px rgba(11,18,48,0.7);
        --placeholder: rgba(230,238,246,0.35);
      }
    }
    body, .stApp { background: var(--bg); color: var(--text); font-family: 'Poppins', system-ui, sans-serif; }
    .aurora-header { text-align: center; margin: 2rem auto 3rem auto; }
    .aurora-logo {
      width: 110px; height: 110px; border-radius: 50%;
      margin: 0 auto 1rem auto;
      background: linear-gradient(135deg, var(--accent1), var(--accent2));
      display:flex; align-items:center; justify-content:center;
      box-shadow: 0 15px 35px rgba(124,77,255,0.35), 0 10px 25px rgba(24,255,255,0.25);
      animation: float 4s ease-in-out infinite;
    }
    .aurora-logo span { font-size: 46px; font-weight: 900; color: white; }
    .aurora-title {
      font-size: 54px; font-weight: 900; margin: 0;
      background: linear-gradient(90deg, var(--accent1), var(--accent2));
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      letter-spacing: 2px; text-transform: uppercase;
    }
    .aurora-sub { margin-top: 12px; font-size: 1.35rem; color: var(--muted); }
    @keyframes float { 0% { transform: translateY(0);} 50% { transform: translateY(-8px);} 100% { transform: translateY(0);} }
    .result, .block {
      background: var(--card); border-radius: var(--radius);
      padding: 24px; font-size: 1.1rem; line-height: 1.65;
      box-shadow: var(--shadow); border: 1px solid var(--card-border);
      backdrop-filter: blur(12px); margin-top: 1rem;
    }
    textarea, input[type="text"], .stTextArea>div>textarea, .stTextInput>div>input {
      border-radius:16px; padding:18px; border:1px solid var(--input-border);
      background: var(--input-bg); color: var(--text); min-height:140px;
      font-family: 'Poppins', sans-serif; font-size: 1.05rem;
    }
    .stButton>button {
      border-radius:14px; padding:12px 24px;
      background-image: linear-gradient(90deg, var(--accent1), var(--accent2));
      color: #fff; font-weight:600; font-size: 1rem; border:none;
      box-shadow: var(--button-shadow);
    }
    .stButton>button:hover { transform: translateY(-2px); }
    [data-testid="stSidebar"] { background: var(--sidebar-bg); padding-top: 1rem; }
    .stTabs [role="tablist"] { gap: 20px; justify-content: center; }
    .stTabs [role="tab"] { border-radius: 12px; padding: 10px 18px; font-weight:600; }
    .stTabs [aria-selected="true"] {
      background: linear-gradient(90deg, var(--accent1), var(--accent2)) !important;
      color:white !important;
    }
    @media (max-width: 768px) { .aurora-title { font-size:32px; } .aurora-logo { width:75px; height:75px; } }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class='aurora-header'>
      <div class='aurora-logo'><span>A</span></div>
      <div>
        <div class='aurora-title'>AURORA</div>
        <div class='aurora-sub'>Where Translation meets Creativity</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def get_hi_bundle():
    return load_translation_bundle(
        "models/besteng2hindi.keras",
        "models/en_tokenizer.pkl",
        "models/hi_tokenizer.pkl"
    )

@st.cache_resource
def get_hinglish_bundle():
    return load_translation_bundle(
        "models/besteng2hinglish.keras",
        "models/eng_tokenizer.pkl",
        "models/hing_tokenizer.pkl"
    )

with st.sidebar:
    st.header("⚙️ Settings")
    target_lang = st.radio("Translate to", ["Hindi", "Hinglish"], index=0)
    st.caption("Pick your preferred language.")

st.subheader("Enter English text to translate: ")
english_text = st.text_area("Input", placeholder="Type English text here...", height=140)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    if st.button("🔁 Translate"):
        if not english_text.strip():
            st.warning("Please enter some English text.")
        else:
            translated = None
            with st.spinner("Translating..."):
                try:
                    resp = requests.post(
                        API_URL,
                        json={"text": english_text, "target_lang": target_lang},
                        timeout=10,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if isinstance(data, dict) and "translation" in data:
                        translated = data["translation"]
                    else:
                        raise ValueError("Invalid API response shape")
                except Exception:
                    if target_lang == "Hindi":
                        model, src_tok, tgt_tok = get_hi_bundle()
                    else:
                        model, src_tok, tgt_tok = get_hinglish_bundle()
                    translated = translate_greedy(model, src_tok, tgt_tok, english_text)

            st.session_state["translated_text"] = translated
            st.session_state["english_text"] = english_text
            st.session_state.pop("generated_text", None)
            st.session_state.pop("styled_text", None)
            st.session_state.pop("adapted_text", None)
            st.session_state.pop("knowledge_text", None)

with col2:
    if st.button("🧹 Clear"):
        st.session_state.clear()

if "translated_text" in st.session_state:
    st.subheader("Translated Text: ")
    st.markdown(f"<div class='result'>{st.session_state['translated_text']}</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Generate Content",
        "🪄 Polish / Transform",
        "🎭 Adapt for Audience",
        "🧠 Enrich with Knowledge",
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            content_type = st.selectbox("Content type", ["Story", "Poem", "Description", "Article", "Ad copy"])
        with c2:
            style = st.selectbox("Style", ["Humor", "Creative", "Interesting", "Formal", "Romantic", "Inspirational"])
        if st.button("✨ Generate", key="generate_btn"):
            try:
                out = query_gemini(st.session_state.get("english_text", ""), st.session_state["translated_text"], content_type, style)
                if not out:
                    st.info("Gemini returned empty text. Try again.")
                else:
                    st.session_state["generated_text"] = out
            except Exception as e:
                st.error(f"Gemini error: {e}")
        if "generated_text" in st.session_state:
            st.markdown("**Output:**")
            st.markdown(f"<div class='block'>{st.session_state['generated_text']}</div>", unsafe_allow_html=True)

    with tab2:
        style_choice = st.selectbox("Choose a style", STYLE_PRESETS, index=0, key="style_polish")
        if st.button("🪄 Rewrite in this style", key="polish_btn"):
            try:
                polished = polish_style(st.session_state["translated_text"], style_choice)
                if polished:
                    st.session_state["styled_text"] = polished
            except Exception as e:
                st.error(f"Gemini error: {e}")
        if "styled_text" in st.session_state:
            st.markdown("**Polished / Transformed:**")
            st.markdown(f"<div class='block'>{st.session_state['styled_text']}</div>", unsafe_allow_html=True)

    with tab3:
        audience = st.selectbox("Audience", AUDIENCE_PRESETS, index=0, key="audience_pick")
        use_styled = "styled_text" in st.session_state and bool(st.session_state["styled_text"])
        if st.button("🎯 Adapt", key="adapt_btn"):
            try:
                source_text = st.session_state["styled_text"] if use_styled else st.session_state["translated_text"]
                adapted = adapt_for_audience(source_text, audience)
                if adapted:
                    st.session_state["adapted_text"] = adapted
            except Exception as e:
                st.error(f"Gemini error: {e}")
        if "adapted_text" in st.session_state:
            st.markdown("**Audience-Adapted Output:**")
            st.markdown(f"<div class='block'>{st.session_state['adapted_text']}</div>", unsafe_allow_html=True)

    with tab4:
        max_items = st.slider("How many suggestions?", 3, 6, 3)
        if st.button("🧠 Enrich with Knowledge", key="knowledge_btn"):
            try:
                enriched = enrich_with_knowledge(st.session_state.get("english_text", ""), st.session_state["translated_text"], max_items=max_items)
                if enriched:
                    st.session_state["knowledge_text"] = enriched
            except Exception as e:
                st.error(f"Enrichment error: {e}")
        if "knowledge_text" in st.session_state:
            st.markdown("**Contextual Suggestions:**")
            st.markdown(f"<div class='block'>{st.session_state['knowledge_text']}</div>", unsafe_allow_html=True)
