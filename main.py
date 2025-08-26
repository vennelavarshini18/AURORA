import os
import streamlit as st
from utils import (
    load_translation_bundle,
    translate_greedy,
    query_gemini,
)

st.set_page_config(page_title="🌐EN → HI/Hinglish + Gemini", layout="wide")

st.markdown("""
<style>
/* simple card look */
.block {padding:1rem;border-radius:12px;border:1px solid #eee;background:#fafafa}
.result {padding:1rem;border-radius:12px;background:#fff;border:1px solid #eaeaea}
</style>
""", unsafe_allow_html=True)

st.title("🌐 English → Hindi/Hinglish Translator  ➜  Gemini Creator ✨")

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

st.subheader("1) Enter English text")
english_text = st.text_area("Input", placeholder="Type English text here...", height=140)

col1, col2 = st.columns([1,1])

with col1:
    if st.button("🔁 Translate"):
        if not english_text.strip():
            st.warning("Please enter some English text.")
        else:
            if target_lang == "Hindi":
                model, src_tok, tgt_tok = get_hi_bundle()
            else:
                model, src_tok, tgt_tok = get_hinglish_bundle()

            translated = translate_greedy(model, src_tok, tgt_tok, english_text)
            st.session_state["translated_text"] = translated

with col2:
    clear = st.button("🧹 Clear")
    if clear:
        st.session_state.pop("translated_text", None)

if "translated_text" in st.session_state:
    st.subheader("2) ✅ Translated Text")
    st.markdown(f"<div class='result'>{st.session_state['translated_text']}</div>", unsafe_allow_html=True)

    st.subheader("3) 🎨 Generate with Gemini")
    c1, c2 = st.columns(2)
    with c1:
        content_type = st.selectbox("Content type", ["Story", "Poem", "Description", "Article", "Ad copy"])
    with c2:
        style = st.selectbox("Style", ["Humor", "Creative", "Interesting", "Formal", "Romantic", "Inspirational"])

    if st.button("✨ Send to Gemini"):
        try:
            out = query_gemini(st.session_state["translated_text"], content_type, style)
            if not out:
                st.info("Gemini returned empty text. Try a different combination.")
            else:
                st.subheader("4) ✨ Gemini Output")
                st.markdown(f"<div class='block'>{out}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Gemini error: {e}")
