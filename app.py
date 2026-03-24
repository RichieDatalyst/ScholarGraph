import os
import asyncio
import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


load_dotenv()

from src.graph import research_graph
from src.tools import ingest_pdf, get_retriever, export_to_pdf
from src.state import AgentState              
from src.agents import chat_with_paper, describe_visuals, generate_comparison_summaries, generate_project_insights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


DATA_DIR       = Path("data")
AUDIO_PATH     = DATA_DIR / "summary_audio.mp3"
EXPERTISE_LEVELS = ["Beginner", "Intermediate", "Expert"]

DATA_DIR.mkdir(exist_ok=True)


st.set_page_config(
    page_title="ScholarGraph: AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# CSS

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] {
        background-color: #1a1d2e;
        border-right: 1px solid #2d3250;
    }
    .summary-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
        border: 1px solid #2d3250;
        border-left: 4px solid #7c83fd;
        border-radius: 12px;
        padding: 28px; margin: 16px 0;
        line-height: 1.8; font-size: 15px; color: #e0e0e0;
    }
    .section-card {
        background: #1a1d2e;
        border: 1px solid #2d3250;
        border-left: 4px solid #51cf66;
        border-radius: 10px;
        padding: 20px; margin: 6px 0;
        line-height: 1.7; font-size: 14px; color: #e0e0e0;
    }
    .audit-card {
        background: #1a1d2e;
        border: 1px solid #ff6b6b44;
        border-left: 4px solid #ff6b6b;
        border-radius: 12px;
        padding: 20px; margin: 12px 0;
        font-size: 14px; color: #e0e0e0;
    }
    .agent-badge {
        display: inline-block;
        background: #7c83fd22; border: 1px solid #7c83fd66;
        border-radius: 20px; padding: 4px 14px;
        font-size: 12px; color: #7c83fd; margin: 4px;
    }
    .section-header {
        font-size: 18px; font-weight: 600;
        color: #7c83fd; margin-bottom: 8px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c83fd, #5b63f5);
        color: white; border: none; border-radius: 8px;
        padding: 10px 24px; font-weight: 600; width: 100%;
    }
    footer { visibility: hidden; }
    .chat-user {
        background: #2d3250;
        border-radius: 12px 12px 4px 12px;
        padding: 12px 16px; margin: 6px 0;
        color: #e0e0e0; font-size: 14px;
    }
    .chat-assistant {
        background: #1a2744;
        border-left: 3px solid #7c83fd;
        border-radius: 4px 12px 12px 12px;
        padding: 12px 16px; margin: 6px 0;
        color: #e0e0e0; font-size: 14px;
        line-height: 1.6;
    }
    .chat-empty {
        text-align: center; padding: 30px;
        color: #555; font-size: 13px;
    }
    .comparison-card {
        background: #1a1d2e;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 20px;
        height: 100%;
        color: #e0e0e0;
        font-size: 14px;
        line-height: 1.7;
    }
    .comparison-header {
        font-size: 15px;
        font-weight: 700;
        color: #7c83fd;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d3250;
    }
    .winner-badge {
        display: inline-block;
        background: #44bb4422;
        border: 1px solid #44bb44;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 11px;
        color: #44bb44;
        margin-left: 8px;
    }        
</style>
""", unsafe_allow_html=True)
 

# SESSION STATE

def init_session_state():
    defaults = {
        "result":           None,
        "audio_ready":      False,
        "last_pdf_name":    None,
        "processing":       False,
        "cached_pdf_name":  None,
        "cached_chunks":    None,
        "cached_retriever": None,
        "cached_raw_text":  None,
        "citations":       [],
        "eval_accuracy":    None,
        "eval_completeness": None,
        "eval_clarity": None,
        "eval_justifications": {},
        "eval_readability_score": None,
        "eval_readability_grade": None,
        "eval_overall": None,
        "chat_history":     [],
        "chat_provider":    "groq",
        "chat_api_key":     "",
        "user_groq_key":   "",
        "user_gemini_key": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
 
init_session_state()
 
# VOICE HELPER

async def _tts_async(text: str, path: str):
    import edge_tts
    await edge_tts.Communicate(text, voice="en-US-AriaNeural").save(path)
 
def generate_voice_summary(text: str) -> bool:
    tts_text = text[:3000] + ("..." if len(text) > 3000 else "")
    try:
        asyncio.run(_tts_async(tts_text, str(AUDIO_PATH)))
        return True
    except RuntimeError:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_tts_async(tts_text, str(AUDIO_PATH)))
            loop.close()
            return True
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return False
 

# SIDEBAR

with st.sidebar:
    st.markdown("## 🔬 ScholarGraph")
    st.markdown("*Your AI-Powered Research Team*")
    st.divider()
 
    st.markdown("### 📄 Upload Paper")
    uploaded_file = st.file_uploader(
        "PDF", type=["pdf"], label_visibility="collapsed"
    )
 
    st.divider()
    st.markdown("### 🎓 Reader Level")
    expertise_level = st.selectbox(
        "Level", EXPERTISE_LEVELS, index=1, label_visibility="collapsed"
    )
    st.caption({
        "Beginner":     "Simple language, analogies, no jargon",
        "Intermediate": "Balanced —> technical terms explained",
        "Expert":       "Full technical depth + ArXiv papers",
    }[expertise_level])


    st.divider()
    # Advanced API key override (collapsed by default)
    with st.expander("Advanced — Use Your Own Keys", expanded=False):
        st.caption(
            "Optional. Leave blank to use the app's built-in keys. "
            "Useful if the app's quota is exhausted."
        )
        user_groq_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_... (free at console.groq.com)",
            key="user_groq_key",
            label_visibility="collapsed",
        )
        st.caption("Groq: [console.groq.com](https://console.groq.com)")

        user_gemini_key = st.text_input(
            "Google API Key",
            type="password",
            placeholder="AIza... (free at aistudio.google.com)",
            key="user_gemini_key",
            label_visibility="collapsed",
        )
        st.caption("Gemini: [aistudio.google.com](https://aistudio.google.com)")

        if user_groq_key or user_gemini_key:
            st.success("Your keys will be used for this session.")
    start_button = st.button(
        "Start Research",
        disabled=uploaded_file is None or st.session_state.processing,
        use_container_width=True,
    )
    if uploaded_file is None:
        st.caption("Upload a PDF to begin.")
 
    # Chat provider settings — only show after a paper is processed
    if st.session_state.result:
        st.divider()
        st.markdown("### Chat Settings")

        provider_options = ["groq", "gemini", "cohere", "cerebras"]
        provider = st.selectbox(
            "LLM Provider",
            provider_options,
            index=provider_options.index(
                st.session_state.get("chat_provider", "groq")
            ),
            label_visibility="collapsed",
        )
        st.session_state.chat_provider = provider

        provider_hints = {
            "groq":   "Free key at console.groq.com",
            "gemini": "Free key at aistudio.google.com",
            "cohere": "Free key at cohere.com",
            "cerebras": "Free key at cloud.cerebras.ai",
        }
        api_key_input = st.text_input(
            "API Key",
            value=st.session_state.get("chat_api_key", ""),
            type="password",
            placeholder=provider_hints[provider],
            label_visibility="collapsed",
        )
        st.session_state.chat_api_key = api_key_input
        st.caption(f"💡 {provider_hints[provider]}")
    
        st.divider()
     
        st.divider()
        st.markdown("### Vision API Key")
        vision_key_input = st.text_input(
            "Google API Key for Vision",
            value=st.session_state.get("vision_api_key", ""),
            type="password",
            placeholder="For visual analysis (aistudio.google.com)",
            label_visibility="collapsed",
        )
        st.session_state.vision_api_key = vision_key_input
        st.caption(
            "💡 Used for chart/figure descriptions. "
            "Free key at [aistudio.google.com](https://aistudio.google.com). "
            "Leave blank to use the app's shared key."
        )

    st.markdown(
        "<div style='font-size:11px;color:#666;text-align:center;'>"
        "Gemini 2.5 Flash Lite · Groq Llama 3.3 · LangGraph · FAISS"
        "</div>", unsafe_allow_html=True
    )
 

# HEADER

st.markdown(
    "<h1 style='color:#7c83fd;margin-bottom:4px;'>🔬 ScholarGraph</h1>"
    "<p style='color:#888;font-size:16px;margin-top:0;'>"
    "An autonomous multi-agent system that reads, reasons, and simplifies "
    "scientific papers as per your expertise level."
    "</p>", unsafe_allow_html=True
)
st.markdown(
    "<div style='display:flex;gap:8px;flex-wrap:wrap;margin:12px 0;'>"
    "<span class='agent-badge'>📥 Ingestion</span>"
    "<span style='color:#666;padding-top:4px;'>→</span>"
    "<span class='agent-badge'>🗺️ Planner</span>"
    "<span style='color:#666;padding-top:4px;'>→</span>"
    "<span class='agent-badge'>🔍 Researcher</span>"
    "<span style='color:#666;padding-top:4px;'>→</span>"
    "<span class='agent-badge'>✍️ Summarizer</span>"
    "<span style='color:#666;padding-top:4px;'>→</span>"
    "<span class='agent-badge'>🔎 Critic</span>"
    "<span style='color:#666;padding-top:4px;'>→</span>"
    "<span class='agent-badge'>📑 Section Summarizer</span>"
    "</div>", unsafe_allow_html=True
)
st.divider()
 

# PROCESSING LOGIC

if start_button and uploaded_file is not None:
    st.session_state.result      = None
    st.session_state.audio_ready = False
    st.session_state.processing  = True
    st.session_state.chat_history = [] 
 
    pdf_path = DATA_DIR / uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info(f"PDF saved to: {pdf_path}")
 
    with st.status("Research agents are working...", expanded=True) as status:
        try:
            # Check cache first to avoid re-processing the same PDF
            if st.session_state.cached_pdf_name == uploaded_file.name:
                st.write("⚡ **Cache Hit** — reusing embeddings for same PDF.")
                chunks    = st.session_state.cached_chunks
                retriever = st.session_state.cached_retriever
                raw_text  = st.session_state.cached_raw_text
                st.write(f"  {len(chunks)} chunks loaded instantly.")
            else:
                st.write("📥 **Ingestion Agent** — Extracting and chunking PDF...")
                chunks = ingest_pdf(str(pdf_path))
                st.write(f"  Extracted {len(chunks)} chunks.")
 
                st.write("🗄️ **Building Knowledge Base** — Local FAISS embeddings...")
                retriever = get_retriever(chunks)
                raw_text  = "\n\n".join(doc.page_content for doc in chunks)
                st.write("   Vector store ready (no API quota used).")
 
                # Cache for this session
                st.session_state.cached_pdf_name  = uploaded_file.name
                st.session_state.cached_chunks    = chunks
                st.session_state.cached_retriever = retriever
                st.session_state.cached_raw_text  = raw_text
 
            st.write("🗺️ **Planner Agent** — Analysing paper structure...")
            st.write("🔍 **Researcher Agent** — Searching knowledge base...")
            if expertise_level == "Expert":
                st.write("   + Fetching related ArXiv papers...")
            st.write(f"✍️ **Summarizer Agent** — Writing {expertise_level}-level summary...")
            st.write("🔎 **Critic Agent** — Fact-checking for hallucinations...")
            st.write("📑 **Section Summarizer** — Breaking down paper section by section...")
 
            # Pass pre-built state to avoid double-processing
            initial_state: AgentState = {
                "pdf_path":              str(pdf_path),
                "user_query":            "",
                "expertise_level":       expertise_level,
                "raw_text":              raw_text,
                "retriever":             retriever,
                "context":               [],
                "arxiv_papers":          [],
                "summary":               "",
                "critic_feedback":       "",
                "is_hallucination_free": False,
                "iteration_count":       0,
                "section_summaries":     None,
                "messages":              [],
                "agent_trace":           [],
            }

            _orig_groq   = os.environ.get("GROQ_API_KEY", "")
            _orig_gemini = os.environ.get("GOOGLE_API_KEY", "")

            if st.session_state.get("user_groq_key"):
                os.environ["GROQ_API_KEY"] = st.session_state.user_groq_key
            if st.session_state.get("user_gemini_key"):
                os.environ["GOOGLE_API_KEY"] = st.session_state.user_gemini_key

            result = research_graph.invoke(initial_state)

            # Restore original keys after run
            os.environ["GROQ_API_KEY"]   = _orig_groq
            os.environ["GOOGLE_API_KEY"] = _orig_gemini
 
            st.session_state.result        = result
            st.session_state.last_pdf_name = uploaded_file.name
            st.session_state.processing    = False
 
            iterations  = result.get("iteration_count", 1)
            is_clear    = result.get("is_hallucination_free", False)
            n_sections  = len(result.get("section_summaries") or {})
 
            status.update(
                label=(
                    f"✅ Research complete! "
                    f"({iterations} iteration(s) · "
                    f"{n_sections} sections found · "
                    f"{'Verified ✓' if is_clear else 'Max iterations reached'})"
                ),
                state="complete",
                expanded=False,
            )
 
        except Exception as e:
            st.session_state.processing = False
            status.update(label="❌ Pipeline failed", state="error", expanded=True)
            st.error(f"**Error:** {e}")
            logger.error(f"Pipeline error: {e}", exc_info=True)
            st.stop()
 

# RESULTS DISPLAY

if st.session_state.result:
    result           = st.session_state.result
    summary          = result.get("summary", "No summary generated.")
    critic_feedback  = result.get("critic_feedback", "")
    arxiv_papers     = result.get("arxiv_papers", [])
    iteration_count  = result.get("iteration_count", 0)
    is_clear         = result.get("is_hallucination_free", False)
    section_summaries = result.get("section_summaries") or {}
 
    col1, col2, col3, col4, col5 = st.columns(5)
    name = st.session_state.last_pdf_name or ""
    col1.metric("📄 Paper",      name[:18] + "..." if len(name) > 18 else name)
    col2.metric("🎓 Level",      result.get("expertise_level", "—"))
    col3.metric("🔁 Iterations", iteration_count)
    col4.metric("✅ Verified",   "Yes" if is_clear else "Forced")
    col5.metric("📑 Sections",   len(section_summaries))
 
    st.divider()
    # Adaptive Summary with TTS and Download
    st.markdown(
        "<div class='section-header'>📋 Adaptive Summary</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='summary-card'>{summary}</div>",
        unsafe_allow_html=True
    )
    voice_col, dl_col = st.columns([1, 1])
    with voice_col:
        if st.button("🎙️ Generate Voice Summary", use_container_width=True):
            with st.spinner("Generating via Edge-TTS..."):
                if generate_voice_summary(summary):
                    st.session_state.audio_ready = True
                else:
                    st.error("TTS failed. Check edge-tts is installed.")
    with dl_col:
        st.download_button(
            "Download Summary (.txt)",
            data=summary,
            file_name=f"scholargraph_{result.get('expertise_level','').lower()}_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )
 
    if st.session_state.audio_ready and AUDIO_PATH.exists():
        with open(AUDIO_PATH, "rb") as af:
            st.audio(af.read(), format="audio/mp3")
 
    st.divider()
 
    # Section-by-section summaries
    st.markdown(
        "<div class='section-header'>📑 Section-by-Section Breakdown</div>",
        unsafe_allow_html=True
    )
 
    if not section_summaries:
        st.info(
            "No standard sections were detected in this paper. "
            "This can happen with non-standard formatting or very short papers."
        )
    else:
        section_icons = {
            "Abstract":     "📄",
            "Introduction": "🔍",
            "Methodology":  "⚙️",
            "Results":      "📊",
            "Discussion":   "💬",
            "Conclusion":   "🏁",
        }
 
        n_detected  = len([k for k in section_summaries
                           if k in ["Abstract", "Introduction", "Related Work",
                                    "Methodology", "Results", "Discussion", "Conclusion"]])
        n_extracted = len(section_summaries) - n_detected
        caption     = f"Found **{len(section_summaries)}** section(s)"
        if n_extracted > 0:
            caption += (f" — {n_detected} detected from paper structure, "
                        f"{n_extracted} extracted from content")
        st.caption(caption)
 
        for section_name, section_text in section_summaries.items():
            icon = section_icons.get(section_name, "📌")
            with st.expander(f"{icon} {section_name}", expanded=False):
                st.markdown(
                    f"<div class='section-card'>{section_text}</div>",
                    unsafe_allow_html=True
                )
 
        # Download all section summaries as one file
        all_sections_text = "\n\n".join(
            f"## {name}\n{text}"
            for name, text in section_summaries.items()
        )
        st.download_button(
            "Download All Section Summaries (.txt)",
            data=all_sections_text,
            file_name="scholargraph_sections.txt",
            mime="text/plain",
        )
 
    st.divider()
 
    # ArXiv Related Papers
    if arxiv_papers:
        st.markdown(
            "<div class='section-header'>📚 Related Papers (ArXiv)</div>",
            unsafe_allow_html=True
        )
        for i, paper in enumerate(arxiv_papers, 1):
            with st.expander(f"📄 Related Paper {i}", expanded=False):
                st.markdown(paper)
        st.divider()

    # Citations 
    citations = result.get("citations") or []
    if citations:
        st.markdown(
            "<div class='section-header'>📚 Extracted Citations</div>",
            unsafe_allow_html=True
        )
        st.caption(f"Found **{len(citations)}** reference(s) in this paper.")
        for cite in citations:
            st.markdown(
                f"<div style='background:#1a1d2e;border-left:3px solid #7c83fd;"
                f"border-radius:6px;padding:10px 14px;margin:4px 0;"
                f"font-size:13px;color:#ccc;'>"
                f"<strong style='color:#7c83fd;'>[{cite['number']}]</strong> "
                f"{cite['text']}</div>",
                unsafe_allow_html=True
            )
        st.divider()

    # Multi-dimensional Evaluation
    eval_accuracy    = result.get("eval_accuracy")
    eval_completeness = result.get("eval_completeness")
    eval_clarity     = result.get("eval_clarity")
    eval_overall     = result.get("eval_overall")
    eval_just        = result.get("eval_justifications") or {}
    eval_flesch      = result.get("eval_readability_score")
    eval_grade       = result.get("eval_readability_grade")

    if eval_accuracy is not None:
        st.markdown(
            "<div class='section-header'> Summary Quality Evaluation</div>",
            unsafe_allow_html=True
        )

        def score_color(s):
            return {1:"#ff4444", 2:"#ff8800", 3:"#ffcc00",
                    4:"#88cc00", 5:"#44bb44"}.get(s, "#888")

        def stars(s):
            return "★" * s + "☆" * (5 - s) if s else "—"

        # Flesch label
        def flesch_label(score):
            if score is None:
                return "—"
            if score >= 70: return f"{score} (Easy)"
            if score >= 50: return f"{score} (Standard)"
            if score >= 30: return f"{score} (Difficult)"
            return f"{score} (Very Difficult)"

        # Build the table rows
        rows = [
            ("📊 Accuracy",      eval_accuracy,
             eval_just.get("Accuracy", "—")),
            ("📋 Completeness",  eval_completeness,
             eval_just.get("Completeness", "—")),
            ("💬 Clarity",       eval_clarity,
             eval_just.get("Clarity", "—")),
        ]

        # Render each row as a styled card
        for label, score, justification in rows:
            color = score_color(score)
            st.markdown(
                f"<div style='background:#1a1d2e;border:1px solid #2d3250;"
                f"border-left:4px solid {color};border-radius:8px;"
                f"padding:14px 18px;margin:6px 0;display:flex;"
                f"align-items:center;gap:16px;'>"
                f"<div style='min-width:140px;font-weight:600;color:#ccc;'>"
                f"{label}</div>"
                f"<div style='min-width:100px;color:{color};font-size:18px;'>"
                f"{stars(score)}</div>"
                f"<div style='min-width:40px;color:{color};font-weight:700;'>"
                f"{score}/5</div>"
                f"<div style='color:#888;font-size:13px;'>{justification}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Readability row
        if eval_flesch is not None:
            flesch_color = score_color(
                5 if eval_flesch >= 70 else
                4 if eval_flesch >= 50 else
                3 if eval_flesch >= 30 else 2
            )
            st.markdown(
                f"<div style='background:#1a1d2e;border:1px solid #2d3250;"
                f"border-left:4px solid {flesch_color};border-radius:8px;"
                f"padding:14px 18px;margin:6px 0;display:flex;"
                f"align-items:center;gap:16px;'>"
                f"<div style='min-width:140px;font-weight:600;color:#ccc;'>"
                f"📖 Readability</div>"
                f"<div style='min-width:100px;color:{flesch_color};"
                f"font-size:15px;font-weight:600;'>"
                f"{flesch_label(eval_flesch)}</div>"
                f"<div style='color:#888;font-size:13px;'>"
                f"Flesch-Kincaid Grade: {eval_grade}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

        # Overall score banner
        if eval_overall:
            overall_color = score_color(round(eval_overall))
            st.markdown(
                f"<div style='background:linear-gradient(135deg,#1a1d2e,#16213e);"
                f"border:1px solid {overall_color}44;"
                f"border-left:4px solid {overall_color};"
                f"border-radius:8px;padding:14px 18px;margin:10px 0;'>"
                f"<span style='color:#aaa;font-size:13px;'>Overall Score</span>"
                f"<span style='color:{overall_color};font-size:24px;"
                f"font-weight:700;margin-left:16px;'>{eval_overall}/5</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.divider() 

    # LLM Comparison
    st.markdown(
        "<div class='section-header'>⚖️ Three-Way LLM Comparison</div>",
        unsafe_allow_html=True
    )
    st.caption(
        "Generate the same summary using three different LLMs and compare "
        "outputs with automatic quality metrics."
    )

    # Cache
    if "comparison_result" not in st.session_state:
        st.session_state.comparison_result = None
    if "comparison_pdf" not in st.session_state:
        st.session_state.comparison_pdf = None
    if st.session_state.comparison_pdf != st.session_state.last_pdf_name:
        st.session_state.comparison_result = None
        st.session_state.comparison_pdf    = None

    # Provider + key inputs
    cmp_c1, cmp_c2, cmp_c3 = st.columns(3)
    provider_opts = ["groq", "cerebras", "gemini", "cohere"]

    with cmp_c1:
        llm_a = st.selectbox("LLM A", provider_opts, index=0, key="cmp_a")
        key_a = st.text_input("Key A", type="password",
                              placeholder="Leave blank = use .env",
                              key="cmp_key_a", label_visibility="collapsed")
    with cmp_c2:
        llm_b = st.selectbox("LLM B", provider_opts, index=1, key="cmp_b")
        key_b = st.text_input("Key B", type="password",
                              placeholder="Leave blank = use .env",
                              key="cmp_key_b", label_visibility="collapsed")
    with cmp_c3:
        llm_c = st.selectbox("LLM C", provider_opts, index=2, key="cmp_c")
        key_c = st.text_input("Key C", type="password",
                              placeholder="Leave blank = use .env",
                              key="cmp_key_c", label_visibility="collapsed")

    run_cmp = st.button("⚖️ Run Three-Way Comparison", use_container_width=False)

    if run_cmp:
        providers = [llm_a, llm_b, llm_c]
        if len(set(providers)) < 3:
            st.warning("Please select three different LLMs.")
        else:
            context  = "\n\n".join(result.get("context") or [])
            raw_text_cmp = result.get("raw_text", "")
            if not context and not raw_text_cmp:
                st.error("No context available.")
            else:
                with st.spinner(
                    f"Generating summaries with {llm_a.capitalize()}, "
                    f"{llm_b.capitalize()}, and {llm_c.capitalize()}..."
                ):
                    try:
                        cmp_data = generate_comparison_summaries(
                            context=context,
                            raw_text=raw_text_cmp,
                            expertise_level=result.get("expertise_level", "Intermediate"),
                            llm_a_provider=llm_a,
                            llm_b_provider=llm_b,
                            llm_c_provider=llm_c,
                            llm_a_key=key_a,
                            llm_b_key=key_b,
                            llm_c_key=key_c,
                            reference_summary=result.get("summary", ""),
                        )
                        st.session_state.comparison_result = cmp_data
                        st.session_state.comparison_pdf    = st.session_state.last_pdf_name
                    except Exception as e:
                        st.error(f"Comparison failed: {e}")

    cmp = st.session_state.get("comparison_result")
    if cmp:
        winner = cmp.get("winner", "N/A")

        # Metric table
        results_list = [cmp["llm_a"], cmp["llm_b"], cmp["llm_c"]]

        st.markdown("**📊 Automatic Quality Metrics**")

        metric_labels = {
            "readability_score":  "Readability (Flesch)",
            "word_count":         "Word Count",
            "lexical_diversity":  "Lexical Diversity",
            "similarity_score":   "Similarity to Pipeline",
            "composite_score":    "Composite Score",
        }

        # Build metric rows
        table_data = {"Metric": list(metric_labels.values())}
        for r in results_list:
            col_name = r["model"]
            m        = r["metrics"]
            table_data[col_name] = [
                f"{m['readability_score']}",
                f"{m['word_count']} words",
                f"{m['lexical_diversity']:.2f}",
                f"{m['similarity_score']:.2f}",
                f"**{m['composite_score']}/10**",
            ]

        import pandas as pd
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Winner banner
        if winner != "N/A":
            winner_model = next(
                (r["model"] for r in results_list if r["provider"] == winner),
                winner
            )
            st.success(f"🏆 **Best performing:** {winner_model} (highest composite score)")

        # ── Three-column summaries ────────────────────────────────────────────
        col_a, col_b, col_c = st.columns(3)
        for col, r in zip([col_a, col_b, col_c], results_list):
            is_winner = r["provider"] == winner
            badge     = " 🏆" if is_winner else ""
            with col:
                st.markdown(
                    f"<div class='comparison-header'>"
                    f" {r['model']}{badge}<br>"
                    f"<span style='color:#888;font-size:11px;'>"
                    f"{r['metrics']['word_count']} words · "
                    f"Score: {r['metrics']['composite_score']}/10"
                    f"</span></div>",
                    unsafe_allow_html=True
                )
                if r.get("error"):
                    st.error(r["summary"])
                else:
                    st.markdown(
                        f"<div class='comparison-card'>{r['summary']}</div>",
                        unsafe_allow_html=True
                    )

        # Preference + download
        st.markdown("**Which summary do you prefer?**")
        pref_cols = st.columns(4)
        for i, r in enumerate(results_list):
            with pref_cols[i]:
                if st.button(f"👍 {r['provider'].capitalize()}",
                             use_container_width=True,
                             key=f"pref_{i}"):
                    st.success(f"You preferred {r['model']}!")
        with pref_cols[3]:
            combined = "\n\n".join(
                f"=== {r['model']} ===\n\n{r['summary']}"
                for r in results_list
            )
            st.download_button(
                "⬇️ Download All",
                data=combined,
                file_name="scholargraph_comparison.txt",
                mime="text/plain",
                use_container_width=True,
            )

    st.divider()

    # Agent Execution Trace 
    agent_trace = result.get("agent_trace") or []
    if agent_trace:
        with st.expander("🔍 Agent Execution Trace", expanded=False):
            st.caption(
                f"Full pipeline trace — {len(agent_trace)} agent steps recorded."
            )
            for step in agent_trace:
                llm_badge = (
                    f"<span style='background:#1a3a1a;border:1px solid #44bb44;"
                    f"border-radius:10px;padding:1px 8px;font-size:10px;"
                    f"color:#44bb44;margin-left:8px;'>{step['llm']}</span>"
                    if step.get("llm") else ""
                )
                st.markdown(
                    f"<div style='background:#1a1d2e;border-left:3px solid #7c83fd;"
                    f"border-radius:4px;padding:10px 14px;margin:4px 0;"
                    f"font-size:13px;color:#ccc;'>"
                    f"<strong style='color:#7c83fd;'>Step {step['step']}</strong>"
                    f"<strong style='color:#ddd;margin-left:8px;'>"
                    f"{step['agent']}</strong>{llm_badge}"
                    f"<br><span style='color:#888;font-size:12px;'>"
                    f"{step['summary']}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
    st.divider()

    # Project Insights Dashboard 
    st.markdown(
        "<div class='section-header'>📈 Project Insights Dashboard</div>",
        unsafe_allow_html=True
    )
    st.caption("Automated evaluation of how well ScholarGraph analysed this paper.")

    raw_text_ins  = result.get("raw_text", "")
    chunk_count   = st.session_state.get("cached_chunks")
    chunk_count   = len(chunk_count) if chunk_count else 0

    insights = generate_project_insights(
        result=result,
        raw_text=raw_text_ins,
        chunk_count=chunk_count,
        pdf_name=st.session_state.last_pdf_name or "",
    )

    overall_score = insights.pop("_overall", 0)

    # Overall score banner
    score_color = (
        "#44bb44" if overall_score >= 8 else
        "#88cc00" if overall_score >= 6 else
        "#ffcc00" if overall_score >= 4 else
        "#ff8800"
    )
    overall_label = (
        "Excellent Analysis" if overall_score >= 8 else
        "Good Analysis"      if overall_score >= 6 else
        "Adequate Analysis"  if overall_score >= 4 else
        "Limited Analysis"
    )
    st.markdown(
        f"<div style='background:linear-gradient(135deg,#1a1d2e,#16213e);"
        f"border:1px solid {score_color}44;"
        f"border-left:4px solid {score_color};"
        f"border-radius:10px;padding:16px 20px;margin:8px 0;'>"
        f"<span style='color:#aaa;font-size:13px;'>Overall Project Score</span>"
        f"<span style='color:{score_color};font-size:28px;font-weight:700;"
        f"margin-left:16px;'>{overall_score}/10</span>"
        f"<span style='color:{score_color};font-size:14px;margin-left:12px;'>"
        f"— {overall_label}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Individual insight cards in a grid
    insight_icons = {
        "paper_complexity":   "📄",
        "summary_quality":    "✍️",
        "section_coverage":   "📑",
        "citations":          "📚",
        "pipeline_efficiency":"⚙️",
        "document_size":      "📊",
    }
    insight_titles = {
        "paper_complexity":   "Paper Complexity",
        "summary_quality":    "Summary Quality",
        "section_coverage":   "Section Coverage",
        "citations":          "Citation Extraction",
        "pipeline_efficiency":"Pipeline Efficiency",
        "document_size":      "Document Size",
    }

    # Display in 3-column grid
    insight_items = list(insights.items())
    for i in range(0, len(insight_items), 3):
        cols = st.columns(3)
        for j, (key, val) in enumerate(insight_items[i:i+3]):
            score  = val["score"]
            max_s  = val["max"]
            label  = val["label"]
            detail = val["detail"]
            icon   = insight_icons.get(key, "📌")
            title  = insight_titles.get(key, key)

            # Score color
            ratio = score / max_s
            color = (
                "#44bb44" if ratio >= 0.8 else
                "#88cc00" if ratio >= 0.6 else
                "#ffcc00" if ratio >= 0.4 else
                "#ff8800"
            )

            # Star display
            filled = int(round(ratio * 5))
            stars  = "★" * filled + "☆" * (5 - filled)

            with cols[j]:
                st.markdown(
                    f"<div style='background:#1a1d2e;border:1px solid #2d3250;"
                    f"border-left:3px solid {color};border-radius:10px;"
                    f"padding:14px;margin:4px 0;height:140px;'>"
                    f"<div style='font-size:12px;color:#888;'>{icon} {title}</div>"
                    f"<div style='color:{color};font-size:18px;margin:4px 0;'>{stars}</div>"
                    f"<div style='color:#ddd;font-size:12px;font-weight:600;'>{label}</div>"
                    f"<div style='color:#666;font-size:11px;margin-top:4px;'>{detail}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    st.divider()   

    # Visual Analysis 
    st.markdown(
        "<div class='section-header'>💠 Visual Analysis</div>",
        unsafe_allow_html=True
    )
    st.caption(
        "Extracts charts, diagrams, tables, and figures from the PDF "
        "and generates AI descriptions using Gemini Vision."
    )

    # Use session state to cache visual results
    if "visual_results" not in st.session_state:
        st.session_state.visual_results = None
    if "visual_pdf_name" not in st.session_state:
        st.session_state.visual_pdf_name = None

    # Reset if new paper loaded
    if st.session_state.visual_pdf_name != st.session_state.last_pdf_name:
        st.session_state.visual_results  = None
        st.session_state.visual_pdf_name = None

    col_vis1, col_vis2 = st.columns([2, 3])
    with col_vis1:
        run_vision = st.button(
            "🔍 Analyse Visuals",
            use_container_width=True,
            help="Extracts all images/charts from the PDF and describes them using Gemini Vision."
        )
    with col_vis2:
        st.caption(
            "⚠️ Requires GOOGLE_API_KEY. "
            "Uses Gemini Vision — separate from the chat provider."
        )

    if run_vision:
        pdf_path = str(DATA_DIR / (st.session_state.last_pdf_name or ""))
        if not os.path.exists(pdf_path):
            st.error("PDF file not found. Please re-upload the paper.")
        else:
            with st.spinner("Extracting and describing visuals..."):
                try:
                    visuals = describe_visuals(
                        pdf_path=pdf_path,
                        expertise_level=result.get("expertise_level", "Intermediate"),
                        api_key=st.session_state.get("vision_api_key", ""),
                    )
                    st.session_state.visual_results  = visuals
                    st.session_state.visual_pdf_name = st.session_state.last_pdf_name
                except Exception as e:
                    st.error(f"Visual analysis failed: {e}")

    # Display results
    visuals = st.session_state.get("visual_results")

    if visuals is not None:
        if not visuals:
            st.info(
                "No images, charts, or figures were found in this PDF. "
                "This is common for text-only papers."
            )
        else:
            st.success(f"Found and described **{len(visuals)}** visual(s).")

            for i, vis in enumerate(visuals, 1):
                caption     = vis.get("caption", f"Visual {i}")
                description = vis.get("description", "No description available.")
                page        = vis.get("page", "?")
                b64_data    = vis.get("base64_data", "")
                media_type  = vis.get("media_type", "image/png")
                width       = vis.get("width", 0)
                height      = vis.get("height", 0)

                with st.expander(
                    f"{'📊' if 'table' in caption.lower() else '💠'} "
                    f"Visual {i} — Page {page}"
                    + (f": {caption[:60]}" if caption else ""),
                    expanded=False
                ):
                    # Display image
                    if b64_data:
                        import base64 as b64
                        img_bytes = b64.b64decode(b64_data)
                        st.image(
                            img_bytes,
                            caption=caption or f"Visual {i} (Page {page})",
                            width="stretch",
                        )

                    # Display description
                    has_vision = vis.get("has_vision", False)
                    if has_vision:
                        st.markdown(
                            f"<div class='section-card'>"
                            f"<strong> AI Vision Description:</strong>"
                            f"<br><br>{description}"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        # Still show caption and paper context
                        # even without AI description
                        st.markdown(
                            f"<div class='section-card' "
                            f"style='border-left-color:#ffcc00;'>"
                            f"{description}"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        st.caption(
                            "💡 AI vision descriptions available when Gemini "
                            "quota is fresh (resets midnight UTC). "
                            "Caption and paper context shown above."
                        )
                    # Metadata
                    st.caption(
                        f"Page {page} · {width}×{height}px"
                        + (f" · {caption}" if caption else "")
                    )

    st.divider()

    # Export to PDF 
    st.markdown(
        "<div class='section-header'>📥 Export Full Report</div>",
        unsafe_allow_html=True
    )
    if st.button("📄 Generate PDF Report", use_container_width=False):
        with st.spinner("Building PDF report..."):
            try:
                
                pdf_path = export_to_pdf(
                    summary=summary,
                    section_summaries=section_summaries,
                    citations=citations,
                    eval_overall=result.get("eval_overall"),
                    eval_accuracy=result.get("eval_accuracy"),
                    eval_completeness=result.get("eval_completeness"),
                    eval_clarity=result.get("eval_clarity"),
                    eval_justifications=result.get("eval_justifications") or {},
                    eval_readability_score=result.get("eval_readability_score"),
                    eval_readability_grade=result.get("eval_readability_grade"),
                    expertise_level=result.get("expertise_level", ""),
                    pdf_name=st.session_state.last_pdf_name or "paper.pdf",
                )
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download PDF Report",
                        data=f.read(),
                        file_name="scholargraph_report.pdf",
                        mime="application/pdf",
                        use_container_width=False,
                    )
                st.success("PDF ready!")
            except ImportError:
                st.error("Run `pip install reportlab` to enable PDF export.")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
    st.divider()    
    
    # Chat with Paper
    st.markdown(
        "<div class='section-header'>💬 Chat with Your Paper</div>",
        unsafe_allow_html=True
    )

    provider     = st.session_state.get("chat_provider", "groq")
    chat_api_key = st.session_state.get("chat_api_key", "")

    # Provider info banner
    provider_info = {
        "groq":   ("Llama 3.3 70B",        "console.groq.com"),
        "gemini": ("Gemini 2.5 Flash Lite", "aistudio.google.com"),
        "cohere": ("Command R+",            "cohere.com"),
    }
    model_name, signup_url = provider_info.get(provider, ("Unknown", ""))

    st.info(
        f"**Active:** {provider.capitalize()} — {model_name}  |  "
        f"Set your API key in the sidebar ←  |  "
        f"Get free key at [{signup_url}](https://{signup_url})",
    )

    # Chat history display 
    chat_history = st.session_state.get("chat_history", [])

    if not chat_history:
        st.markdown(
            "<div class='chat-empty'>"
            "Ask any question about the paper above ↑<br>"
            "<small>e.g. What datasets were used? "
            "What are the main limitations? "
            "Explain the methodology simply.</small>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        for turn in chat_history:
            if turn["role"] == "user":
                st.markdown(
                    f"<div class='chat-user'>"
                    f"👤 <strong>You:</strong> {turn['content']}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                # Render assistant response as markdown for formatting
                st.markdown(
                    f"<div class='chat-assistant'>"
                    f"🤖 <strong>Assistant:</strong><br>{turn['content']}"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # Chat input 
    chat_col1, chat_col2 = st.columns([5, 1])
    with chat_col1:
        chat_input = st.text_input(
            "Ask a question about this paper",
            placeholder="e.g. What problem does this paper solve?",
            label_visibility="collapsed",
            key="chat_input_field",
        )
    with chat_col2:
        send_btn = st.button("Send ➤", use_container_width=True)

    if send_btn and chat_input.strip():

        # Validate API key before calling
        if not chat_api_key.strip():
            # Try env key as fallback
            env_keys = {
                "groq":   os.getenv("GROQ_API_KEY", ""),
                "gemini": os.getenv("GOOGLE_API_KEY", ""),
                "cohere": os.getenv("COHERE_API_KEY", ""),
            }
            if not env_keys.get(provider, ""):
                st.error(
                    f"Please enter your {provider.capitalize()} API key "
                    f"in the sidebar to use the chatbot."
                )
                st.stop()

        retriever = st.session_state.get("cached_retriever")

        with st.spinner(f"Thinking via {provider.capitalize()}..."):
            response = chat_with_paper(
                user_message=chat_input.strip(),
                retriever=retriever,
                chat_history=st.session_state.chat_history,
                expertise_level=result.get("expertise_level", "Intermediate"),
                provider=provider,
                api_key=chat_api_key,
            )

        # Append to history
        st.session_state.chat_history.append(
            {"role": "user",      "content": chat_input.strip()}
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )
        st.rerun()

    # Clear chat button — only show if there's history
    if chat_history:
        if st.button("🗑️ Clear Chat History", use_container_width=False):
            st.session_state.chat_history = []
            st.session_state.visual_results  = None   
            st.session_state.visual_pdf_name = None
            st.rerun()


    # Critic Audit Log 
    st.markdown(
        "<div class='section-header'>🔎 Critic Audit Log</div>",
        unsafe_allow_html=True
    )
    if not critic_feedback or critic_feedback.strip().upper() == "CLEAR":
        st.markdown(
            "<div class='audit-card' style='border-left-color:#51cf66;'>"
            "✅ <strong>CLEAR</strong> — Summary passed hallucination review. "
            "All claims are supported by the source document."
            "</div>", unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='audit-card'>⚠️ <strong>Critic Feedback:</strong>"
            f"<br><br>{critic_feedback}</div>",
            unsafe_allow_html=True
        )
 
# Empty State
elif not st.session_state.result:
    st.markdown(
        "<div style='text-align:center;padding:60px 20px;color:#555;'>"
        "<div style='font-size:64px;'>🔬</div>"
        "<h3 style='color:#666;'>Upload a research paper to begin</h3>"
        "<p>ScholarGraph will analyze it using a team of AI agents and deliver<br>"
        "a personalized summary customized to your expertise level.</p>"
        "<br>"
        "<div style='display:flex;justify-content:center;gap:32px;flex-wrap:wrap;'>"
        "<div><strong style='color:#7c83fd;'>📥</strong><br>Upload PDF</div>"
        "<div><strong style='color:#7c83fd;'>→</strong></div>"
        "<div><strong style='color:#7c83fd;'></strong><br>Agents Analyse</div>"
        "<div><strong style='color:#7c83fd;'>→</strong></div>"
        "<div><strong style='color:#7c83fd;'>📋</strong><br>Overall Summary</div>"
        "<div><strong style='color:#7c83fd;'>→</strong></div>"
        "<div><strong style='color:#7c83fd;'>📑</strong><br>Section Summaries</div>"
        "<div><strong style='color:#7c83fd;'>→</strong></div>"
        "<div><strong style='color:#7c83fd;'>🔊</strong><br>Listen to it</div>"
        "</div>"
        "</div>", unsafe_allow_html=True
    )