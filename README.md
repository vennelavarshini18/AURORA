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

Aurora goes beyond simple translation by integrating **Google Gemini** with **LangChain’s agent + tool ecosystem**, enabling powerful, controlled workflows that combine generation, retrieval, and safe tool usage.

#### 1. Style Transfer & Audience Adaptation
- Apply **controlled style transfer** (e.g., humorous, poetic, formal) and **audience-specific adaptation** (children, corporate leaders, peers).
- Uses **prompt engineering**, **instruction-following**, and **human-in-the-loop controls** (presets for tone, length, and script enforcement).
- Core concepts: *Conditional generation*, *Prompt conditioning*, *Style presets*.

#### 2. Polish & Refine Writing
- Improve grammar, structure, and fluency to produce professional-grade output.
- Enforces **guardrails** (script enforcement: Hinglish vs. Hindi), **structured outputs**, and **constrained generation** (max-words, output-only rules).
- Core concepts: *Controlled generation*, *Output constraints*, *Quality refinement*.

#### 3. Knowledge Enrichment & Grounding
- Dynamically fetch **trusted contextual knowledge** via Wikipedia / DuckDuckGo when helpful.
- LangChain **agents** decide whether to call **tools** (search, wiki, keyword extraction, math) and synthesize results.
- This creates **Retrieval-Augmented Generation (RAG)** workflows that **ground** the LLM and reduce hallucinations.
- Core concepts: *Agents*, *Tools*, *Answer grounding*, *RAG*.

#### 4. Safe & Controlled Reasoning
- Handle arithmetic and small computations via a **sandboxed math evaluator** (no arbitrary `eval`), ensuring execution safety.
- Uses **routing & fallback**: prefer tool-augmented, grounded responses; fall back to Gemini generation if tools are not useful.
- Core concepts: *Sandboxing*, *Safe execution*, *Routing / Fallback strategies*.

### Core Concepts Integrated (At-a-glance)
- **LangChain**: LLM Wrappers, Agents, Tools, Answer Grounding, RAG, Routing/Fallback, Structured Outputs.  
- **Gemini (GenAI SDK)**: Generative Models, Prompt Engineering, Controlled Style Transfer, Instruction Following, Script Enforcement, Sandboxing.  
- **Cross-cutting**: Guardrails, Human-in-the-loop Controls, Knowledge Enrichment, Hybrid Retrieval + Generation.

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
