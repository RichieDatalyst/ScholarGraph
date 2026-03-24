# 🔬 ScholarGraph —> Autonomous Multi-Agent Research Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.60-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **ScholarGraph** is an autonomous, multi-agent research ecosystem that transforms complex scientific papers into personalized, adaptive insights —> supporting Beginner, Intermediate, and Expert readers with conversational AI, visual analysis, and structured summaries.

---

## Live Demo
**Access the App here:** [scholargraphai.streamlit.app](https://scholargraphai.streamlit.app)

## Demo Link


## Problem Statement

Scientific research is advancing rapidly, but technical papers remain inaccessible to non-experts due to specialized jargon and complex structures. ScholarGraph bridges this gap by acting as a digital "Research Team" that parses, reasons, and simplifies complex scientific documents into personalized insights tailored to the reader's expertise level.

---

## 🏗️ System Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Pipeline                        │
│                                                             │
│  📥 Ingestion  →  🗺️ Planner  →  🔍 Researcher             │
│  (PyMuPDF)        (Gemini/Groq)   (FAISS RAG + ArXiv)       │
│                                        │                    │
│                                        ▼                    │
│                              ✍️ Summarizer Agent            │
│                              (Adaptive: B/I/E level)        │
│                                        │                    │
│                                        ▼                    │
│                              🔎 Critic Agent  ←──────┐      │
│                              (Hallucination Check)   │      │
│                                        │             │      │
│                                   CLEAR?       Issues│      │
│                                        │             │      │
│                                       YES ────────────      │
│                                        │                    │
│                                        ▼                    │
│                              📑 Section Summarizer          │
│                              (Layer 1: Detected)            │
│                              (Layer 2: Critical Extraction) │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit UI Features                     │
│                                                             │
│  📋 Adaptive Summary      📑 Section Summaries              │
│  🎯 Quality Evaluation    📚 Citation Extraction            │
│  🖼️ Visual Analysis       💬 Conversational Chatbot         │
│  🔊 Voice Synthesis       📥 PDF Report Export              │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Adaptive Summaries** | Three expertise levels: Beginner (analogies), Intermediate (balanced), Expert (technical) |
| **Multi-Agent Pipeline** | LangGraph orchestrates Planner → Researcher → Summarizer → Critic loop |
| **Hallucination Detection** | Critic agent cross-checks every summary against source — loops until CLEAR |
| **Section-wise Summaries** | Detects paper structure + extracts 5 guaranteed critical sections |
| **Quality Evaluation** | LLM-judged Accuracy, Completeness, Clarity + Flesch-Kincaid readability |
| **Citation Extraction** | Automatically parses reference lists (IEEE, APA, numbered styles) |
| **Visual Analysis** | Extracts charts/figures from PDF + Gemini Vision descriptions |
| **Conversational RAG** | Chat with your paper using Groq, Gemini, or Cohere |
| **Voice Synthesis** | Edge-TTS reads summaries aloud (no API key needed) |
| **PDF Export** | Professional report with evaluation table, sections, and citations |
| **Zero Embedding Cost** | Local sentence-transformers — no embedding API quota ever consumed |
| **Smart Fallback** | Gemini → Groq automatic switching on quota exhaustion |

---

## 🛠️ Tech Stack (Zero-Cost)

| Component | Technology |
|---|---|
| **LLM (Primary)** | Google Gemini 2.5 Flash Lite (Free via Google AI Studio) |
| **LLM (Fallback)** | Groq — Llama 3.3 70B (Free, ~2s response) |
| **Orchestration** | LangGraph (Multi-agent StateGraph) |
| **Embeddings** | sentence-transformers BAAI/bge-small-en-v1.5 (Local, FREE) |
| **Vector Store** | FAISS (Local) |
| **PDF Processing** | PyMuPDF + pymupdf4llm (Markdown extraction) |
| **Vision LLM** | Gemini 3.1 Flash Lite Preview (Image/chart descriptions) |
| **UI** | Streamlit |
| **Voice** | Edge-TTS (Local, free) |
| **PDF Export** | ReportLab (Local) |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ScholarGraph.git
cd ScholarGraph
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_key_here
GROQ_API_KEY=your_groq_key_here
```

Get free keys:
- Gemini: [aistudio.google.com](https://aistudio.google.com)
- Groq: [console.groq.com](https://console.groq.com)

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
ScholarGraph/
├── app.py                  # Streamlit UI (main entry point)
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed to Git)
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── state.py            # LangGraph AgentState TypedDict
│   ├── graph.py            # Pipeline orchestration + LangGraph StateGraph
│   ├── agents.py           # All agent logic (Planner, Researcher, Summarizer, Critic, etc.)
│   └── tools.py            # PDF extraction, FAISS, RAG, citation extraction
└── data/                   # Temporary PDF storage (gitignored)
```

---

## 🔑 API Keys Required

| Key | Required | Free Tier | Get It |
|---|---|---|---|
| `GOOGLE_API_KEY` | Yes (primary LLM + vision) | 1500 RPD | [aistudio.google.com](https://aistudio.google.com) |
| `GROQ_API_KEY` | Yes (fallback LLM + chat) | 14,400 RPD | [console.groq.com](https://console.groq.com) |
| `COHERE_API_KEY` | Optional (chat only) | 20 trial calls | [cohere.com](https://cohere.com) |

---

## 📊 Evaluation Methodology

ScholarGraph evaluates every generated summary on four dimensions:

| Dimension | Method | Scale |
|---|---|---|
| **Accuracy** | LLM-as-judge: claims vs. source context | 1–5 |
| **Completeness** | LLM-as-judge: key points covered | 1–5 |
| **Clarity** | LLM-as-judge: appropriate for expertise level | 1–5 |
| **Readability** | Flesch-Kincaid (local, no LLM) | Grade level |

Overall score = average of Accuracy + Completeness + Clarity.

---

## 🔄 Agent Pipeline Details

### Hallucination Control Loop
```
Summarizer → Critic → CLEAR? → Section Summarizer
                ↑        NO  ↓
                └──── Revise (max 3 iterations)
```

### Section Detection Strategy
- **Pass 1**: Match headings against 8 known academic patterns
- **Pass 2**: Extract any heading-like line if < 3 known sections found
- **Layer 2**: LLM extracts 5 guaranteed critical sections from full text

### LLM Fallback Chain
```
Gemini 2.5 Flash Lite
    ↓ (on 429 quota exhausted)
Groq Llama 3.3 70B
    ↓ (on failure)
Error message with instructions
```

---

## 🎓 Academic Context

This project extends and significantly upgrades a previous NLP project on adaptive scientific paper summarization. Key improvements:

| Previous Project | ScholarGraph |
|---|---|
| Static T5 summarizer | Multi-agent LangGraph pipeline |
| Single summary output | Adaptive + Section-wise summaries |
| No hallucination check | Critic agent with revision loop |
| Google/Wikipedia APIs | Local FAISS + ArXiv RAG |
| No evaluation beyond metrics | LLM-judged + readability evaluation |
| No interactivity | Conversational chatbot |
| No visual analysis | Gemini Vision for charts/figures |

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Ameer Abdullah**
Department of Data Science, FAST-NUCES Lahore
[abdullahameer255@gmail.com](mailto:abdullahameer255@gmail.com)

---

*Built with LangGraph, Streamlit, Gemini, and Groq — entirely on free-tier APIs.*
