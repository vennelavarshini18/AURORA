import os
import streamlit as st
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

st.set_page_config(page_title="🌐EN → HI/Hinglish + Gemini", layout="wide")

st.markdown("""
<style>
.block {padding:1rem;border-radius:12px;border:1px solid #eee;background:#fafafa}
.result {padding:1rem;border-radius:12px;background:#fff;border:1px solid #eaeaea}
.subtle {color:#666;font-size:0.9rem}
hr {border:none;border-top:1px solid #eee;margin:1.25rem 0}
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
            st.session_state["english_text"] = english_text
            st.session_state.pop("generated_text", None)
            st.session_state.pop("styled_text", None)
            st.session_state.pop("adapted_text", None)
            st.session_state.pop("knowledge_text", None)

with col2:
    if st.button("🧹 Clear"):
        st.session_state.clear()

if "translated_text" in st.session_state:
    st.subheader("2) ✅ Translated Text")
    st.markdown(f"<div class='result'>{st.session_state['translated_text']}</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Generate Content",
        "🪄 Polish / Transform",
        "🎭 Adapt for Audience",
        "🧠 Enrich with Knowledge",
    ])

    with tab1:
        st.markdown("<div class='subtle'>Turn your translation into creative forms while keeping language variant consistent.</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            content_type = st.selectbox("Content type", ["Story", "Poem", "Description", "Article", "Ad copy"])
        with c2:
            style = st.selectbox("Style", ["Humor", "Creative", "Interesting", "Formal", "Romantic", "Inspirational"])
        if st.button("✨ Generate"):
            try:
                out = query_gemini(st.session_state.get("english_text", ""), st.session_state["translated_text"], content_type, style)
                if not out:
                    st.info("Gemini returned empty text. Try a different combination.")
                else:
                    st.session_state["generated_text"] = out
            except Exception as e:
                st.error(f"Gemini error: {e}")
        if "generated_text" in st.session_state and st.session_state["generated_text"]:
            st.markdown("**Output:**")
            st.markdown(f"<div class='block'>{st.session_state['generated_text']}</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='subtle'>Improve fluency or transform tone without changing meaning.</div>", unsafe_allow_html=True)
        style_choice = st.selectbox("Choose a style", STYLE_PRESETS, index=0, key="style_polish")
        if st.button("🪄 Rewrite in this style"):
            try:
                src_text = st.session_state["translated_text"]
                polished = polish_style(src_text, style_choice)
                if polished:
                    st.session_state["styled_text"] = polished
                else:
                    st.info("No text returned. Try another style.")
            except Exception as e:
                st.error(f"Gemini error: {e}")
        if "styled_text" in st.session_state and st.session_state["styled_text"]:
            st.markdown("**Polished / Transformed:**")
            st.markdown(f"<div class='block'>{st.session_state['styled_text']}</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='subtle'>Retarget the message for a specific audience (boss, kids, partner, etc.).</div>", unsafe_allow_html=True)
        audience = st.selectbox("Audience", AUDIENCE_PRESETS, index=0, key="audience_pick")
        use_styled = "styled_text" in st.session_state and bool(st.session_state["styled_text"])
        src_label = "Polished text" if use_styled else "Translated text"
        st.caption(f"Input source: {src_label}")
        if st.button("🎯 Adapt"):
            try:
                source_text = st.session_state["styled_text"] if use_styled else st.session_state["translated_text"]
                adapted = adapt_for_audience(source_text, audience)
                if adapted:
                    st.session_state["adapted_text"] = adapted
                else:
                    st.info("No text returned. Try another audience.")
            except Exception as e:
                st.error(f"Gemini error: {e}")
        if "adapted_text" in st.session_state and st.session_state["adapted_text"]:
            st.markdown("**Audience-Adapted Output:**")
            st.markdown(f"<div class='block'>{st.session_state['adapted_text']}</div>", unsafe_allow_html=True)

    with tab4:
        max_items = st.slider("How many suggestions?", min_value=3, max_value=6, value=3)
        if st.button("🧠 Enrich with Knowledge"):
            try:
                enriched = enrich_with_knowledge(st.session_state.get("english_text", ""), st.session_state["translated_text"], max_items=max_items)
                if enriched:
                    st.session_state["knowledge_text"] = enriched
                else:
                    st.info("No knowledge suggestions returned.")
            except Exception as e:
                st.error(f"Gemini error: {e}")
        if "knowledge_text" in st.session_state and st.session_state["knowledge_text"]:
            st.markdown("**Contextual Suggestions:**")
            st.markdown(f"<div class='block'>{st.session_state['knowledge_text']}</div>", unsafe_allow_html=True)
