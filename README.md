# 🌌 Aurora – Creative Translation & Transformation App  

## The Problem  
Most modern translation and text generation systems rely heavily on **pretrained models or transfer learning**. While powerful, these approaches often produce verbose, generic, or contextually mismatched results, especially when trained on . They also focus only on **literal translation** rather than capturing the **essence and meaning** of the input.
Moreover, users often need **more than just translation**: seamlessly within one workflow.  

## How Aurora solves it
**Aurora** is not just a translation app—it is a **creative transformation platform**.  
It takes simple English input and:  
- Translates into **Hindi or Hinglish** (based on user preference).  
- Transforms text into **poems, stories, humorous notes, or motivational pieces**.  
- Polishes style to make writing **concise, elegant, or expressive**.  
- Adapts output for specific audiences (e.g., kids, boss, students).  
- Enriches text with **relevant external knowledge** from trusted sources using **LangChain Tools** (Wikipedia, DuckDuckGo, etc.).  

Aurora combines **traditional deep learning** with **modern generative AI**, providing a **layered workflow** where translation is only the first step toward creativity.  

## Core Technology  

### Translation Model (Built From Scratch)  
Aurora includes a **custom translation model** trained with **TensorFlow-Keras** from scratch on ~50k rows:  
- **Encoder-only Transformer architecture** (instead of full encoder-decoder).  
- **Stopwords removal** improves learning efficiency for smaller datasets.  
- Focuses on capturing **meaningful embeddings** → outputs are **clear, concise, and meaning-preserving**.  
- Outperformed traditional Transformer pipelines on small-scale bilingual data.  

### Generative AI Layer (Gemini + LangChain)  
Beyond translation, Aurora integrates **Google Gemini** + **LangChain Tools** to enable creative workflows:  
1. **Style Transfer** – Rewrite text in different tones (e.g., humorous, inspirational, poetic). 
2. **Polish Writing** – Refine grammar, structure, and flow for professional output.  
3. **Audience Adaptation** – Personalize text for specific audiences like children, corporate leaders, or peers.  
4. **Knowledge Enrichment** – Dynamically fetch **trusted contextual knowledge** via Wikipedia/DuckDuckGo (through LangChain).  

This hybrid design makes Aurora stand out as both **functional and creative**.

## Tech Stack  
- **Frontend:** [Streamlit](https://streamlit.io/)  
- **Core ML Model:** TensorFlow-Keras (Encoder-only Transformer)  
- **Generative AI:** Gemini API  
- **Knowledge Tools:** LangChain + Wikipedia/DuckDuckGo integrations  
- **Deployment:** Streamlit Cloud

 ## Vision  
Aurora is not just a translator but a **creative AI companion**.  
It aims to break the barrier between **language translation** and **imaginative transformation**, making it possible for anyone to turn simple text into **impactful, knowledge-rich, audience-ready content in preferred language**—all in one platform.  
