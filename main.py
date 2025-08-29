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

st.set_page_config(page_title="AURORA", layout="wide")

st.markdown(
    """
    <style>
    :root {
      --bg: #f6f7fb;
      --card: #ffffff;
      --muted: #6b7280;
      --text: #0f172a;
      --accent1: #7c4dff;
      --accent2: #3dd1d6;
      --radius: 16px;
      --input-bg: #ffffff;
      --input-border: rgba(15,23,42,0.08);
      --shadow: 0 8px 22px rgba(16,24,40,0.05);
      --card-border: rgba(15,23,42,0.06);
      --sidebar-bg: linear-gradient(180deg, #fff, #f9f9ff);
      --button-shadow: 0 8px 22px rgba(124,77,255,0.15);
      --placeholder: rgba(15,23,42,0.35);
    }

    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #071026;
        --card: #0f1720;
        --muted: #9ca3af;
        --text: #e6eef6;
        --accent1: #8b5cf6;
        --accent2: #06b6d4;
        --radius: 16px;
        --input-bg: #0b0f14;
        --input-border: rgba(255,255,255,0.04);
        --shadow: 0 10px 30px rgba(2,6,23,0.7);
        --card-border: rgba(255,255,255,0.04);
        --sidebar-bg: linear-gradient(180deg, #071025, #081127);
        --button-shadow: 0 8px 30px rgba(11,18,48,0.6);
        --placeholder: rgba(230,238,246,0.35);
      }
    }

    /* Fallback for Streamlit internal dark-theme marker (some builds) */
    [data-theme="dark"] :root,
    body[data-theme="dark"] :root {
      /* the @media will usually handle this, but keep as a fallback */
    }

    /* Page / global */
    body, .stApp {
      background: var(--bg);
      color: var(--text);
      font-family: 'Poppins', system-ui, sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    /* Header block */
    .aurora-header {
      display:flex;
      gap:18px;
      align-items:center;
      margin-bottom: 2rem;
      padding: 12px 0;
    }
    .aurora-logo {
      width:72px;
      height:72px;
      border-radius:18px;
      background: linear-gradient(135deg,var(--accent1),var(--accent2));
      display:flex;
      align-items:center;
      justify-content:center;
      color:white;
      font-weight:700;
      box-shadow: 0 10px 32px rgba(68,56,255,0.15);
      font-size:32px;
    }
    .aurora-title {
      font-size:34px;
      font-weight:800;
      margin:0;
      background: linear-gradient(90deg, var(--accent1), var(--accent2));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .aurora-sub {
      color:var(--muted);
      margin-top:6px;
      font-size:1.05rem;
      font-weight:400;
    }

    /* Section titles */
    .stSubheader, .stMarkdown h2 {
      font-size: 1.5rem;
      font-weight: 700;
      margin-top: 2rem;
      margin-bottom: 1rem;
      color: var(--text) ;
    }

    /* Cards */
    .result, .block {
      background:var(--card);
      border-radius: var(--radius);
      padding:22px;
      font-size:1.05rem;
      line-height:1.6;
      box-shadow: var(--shadow);
      border: 1px solid var(--card-border);
      margin-top: 0.8rem;
      color: var(--text);
    }

    /* Inputs */
    textarea, input[type="text"], .stTextArea>div>textarea, .stTextInput>div>input {
      border-radius:14px;
      padding:18px;
      box-shadow:none;
      border:1px solid var(--input-border);
      background: var(--input-bg);
      color: var(--text);
      resize: vertical;
      min-height:140px;
      font-family: 'Poppins', sans-serif;
      font-size: 1.05rem;
      line-height:1.6;
    }
    /* placeholder color */
    textarea::placeholder, input::placeholder {
      color: var(--placeholder);
      opacity: 1;
    }

    /* Buttons (global) */
    .stButton>button, button {
      border-radius:14px;
      padding:12px 20px;
      background-image: linear-gradient(90deg,var(--accent1),var(--accent2));
      color: #fff;
      font-weight:600;
      font-size: 0.95rem;
      border:none;
      box-shadow: var(--button-shadow);
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton>button:hover {
      transform: translateY(-2px);
    }

    /* Secondary button (clear) */
    .clear-btn {
      background: transparent;
      color: var(--accent1);
      border:1px solid rgba(124,77,255,0.18);
      box-shadow:none;
    }

    .subtle { color: var(--muted); font-size:0.95rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
      background: var(--sidebar-bg);
      border-right: 1px solid rgba(15,23,42,0.05);
      padding-top: 1rem;
      color: var(--text);
    }
    [data-testid="stSidebar"] * {
      color: var(--text);
      fill: var(--text);
    }
    [data-testid="stSidebar"] .stButton>button { width:100%; }

    /* Tabs */
    .stTabs [role="tablist"] {
      gap:20px;
      justify-content:center;
    }
    .stTabs [role="tab"] {
      border-radius: 12px;
      padding: 10px 18px;
      font-weight:600;
      font-size:1.05rem;
      color: var(--text) !important;
      background: transparent;
      border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
      background: linear-gradient(90deg, var(--accent1), var(--accent2)) !important;
      color:white !important;
      box-shadow: var(--button-shadow);
    }

    /* Translated text block improvements */
    .translated-block {
      margin-top: 2rem;
      margin-bottom: 2rem;
    }
    .translated-block textarea {
      min-height: 140px;
      font-size: 1.1rem;
      padding: 18px;
      border-radius: 16px;
      box-shadow: var(--shadow);
      background: var(--card);
      color: var(--text);
      border: 1px solid var(--card-border);
    }

    /* Dropdown row */
    .dropdown-row {
      display: flex;
      gap: 20px;
      margin-top: 1.2rem;
      margin-bottom: 1.5rem;
    }
    .dropdown-row .stSelectbox {
      flex: 1;
    }

    /* Action buttons inside tabs → vertical spacing */
    .stTabs .stButton>button {
      margin-top: 1.5rem;
      margin-bottom: 1.5rem;
    }

    /* Responsiveness */
    @media (max-width: 768px) {
      .aurora-title { font-size:26px; }
      .aurora-logo { width:56px; height:56px; font-size:22px; }
      .dropdown-row { flex-direction: column; }
    }

    /* Minor fixes for Streamlit internal class differences */
    /* Make cards inside expanders and other widgets inherit our card background */
    .css-1v3fvcr, .css-1d391kg {
      background: transparent !important;
      color: var(--text) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='aurora-header'>
      <div class='aurora-logo'>A</div>
      <div>
        <div class='aurora-title'>AURORA <span style='font-size:20px;opacity:0.95'>✨</span></div>
        <div class='aurora-sub'>Instant, natural translations — with creative transforms</div>
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
        st.markdown("<div class='subtle'>Turn your translation into creative forms in your preferred language. </div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            content_type = st.selectbox("Content type", ["Story", "Poem", "Description", "Article", "Ad copy"])
        with c2:
            style = st.selectbox("Style", ["Humor", "Creative", "Interesting", "Formal", "Romantic", "Inspirational"])
        if st.button("✨ Generate", key="generate_btn", help="Generate content", type="primary"):
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
        if st.button("🪄 Rewrite in this style", key="polish_btn"):
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
        if st.button("🎯 Adapt", key="adapt_btn"):
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
        if st.button("🧠 Enrich with Knowledge", key="knowledge_btn"):
            try:
                enriched = enrich_with_knowledge(
                    st.session_state.get("english_text", ""),
                    st.session_state["translated_text"],
                    max_items=max_items
                )
                if enriched:
                    st.session_state["knowledge_text"] = enriched
                else:
                    st.info("No knowledge suggestions returned.")
            except Exception as e:
                st.error(f"Enrichment error: {e}")
        if "knowledge_text" in st.session_state and st.session_state["knowledge_text"]:
            st.markdown("**Contextual Suggestions:**")
            st.markdown(f"<div class='block'>{st.session_state['knowledge_text']}</div>", unsafe_allow_html=True)
