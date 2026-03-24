import os
import logging
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.agents import (
    planner_agent,
    researcher_agent,
    summarizer_agent,
    critic_agent,
    section_summarizer_agent,
)
from src.tools import ingest_pdf, get_retriever


load_dotenv()
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
 
MAX_ITERATIONS = 3
 
import src.agents as _agents_module
_agents_module._agent_trace = [] 

# INGESTION NODE
 
def ingestion_node(state: AgentState) -> dict:
    """
    PDF → Markdown → FAISS.
    Skips if raw_text + retriever already in state.
    """
    logger.info("Ingestion node started.")
 
    if state.get("raw_text") and state.get("retriever") is not None:
        logger.info("Ingestion: pre-filled state — skipping duplicate processing.")
        return {}
 
    pdf_path = state.get("pdf_path", "")
    if not pdf_path or not os.path.exists(pdf_path):
        logger.error(f"PDF not found: '{pdf_path}'")
        return {"raw_text": "", "retriever": None}
 
    try:
        chunks    = ingest_pdf(pdf_path)
        retriever = get_retriever(chunks)
        raw_text  = "\n\n".join(doc.page_content for doc in chunks)
        logger.info(f"Ingestion: {len(chunks)} chunks, {len(raw_text):,} chars.")
        return {"raw_text": raw_text, "retriever": retriever}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return {"raw_text": "", "retriever": None}
 
# ROUTING LOGIC
 
def should_continue(state: AgentState) -> str:
    """
    Routes after critic:
        CLEAR  → section_summarizer → END
        Issues → back to summarizer (max 3 loops)
    """
    iteration             = state.get("iteration_count", 0)
    is_hallucination_free = state.get("is_hallucination_free", False)
    critic_feedback       = state.get("critic_feedback", "")
 
    if iteration >= MAX_ITERATIONS:
        logger.warning(f"Router: MAX_ITERATIONS reached. Forcing exit.")
        return "end"
 
    if is_hallucination_free or critic_feedback.strip().upper() == "CLEAR":
        logger.info("Router: CLEAR — routing to section summarizer.")
        return "end"
 
    logger.info(f"Router: Issues (iteration {iteration}). Looping.")
    return "continue"
 
# GRAPH CONSTRUCTION
 
def build_graph() -> StateGraph:
    """Construct and compile the full ScholarGraph pipeline."""
    logger.info("Building ScholarGraph StateGraph...")
 
    workflow = StateGraph(AgentState)
 
    # Nodes
    workflow.add_node("ingestion",          ingestion_node)
    workflow.add_node("planner",            planner_agent)
    workflow.add_node("researcher",         researcher_agent)
    workflow.add_node("summarizer",         summarizer_agent)
    workflow.add_node("critic",             critic_agent)
    workflow.add_node("section_summarizer", section_summarizer_agent)
 
    # Entry point
    workflow.set_entry_point("ingestion")
 
    # Define edges (linear flow)
    workflow.add_edge("ingestion",          "planner")
    workflow.add_edge("planner",            "researcher")
    workflow.add_edge("researcher",         "summarizer")
    workflow.add_edge("summarizer",         "critic")
    workflow.add_edge("section_summarizer", END)
 
    # Conditional edge from critic back to summarizer or forward to section_summarizer based on feedback and iteration count
    workflow.add_conditional_edges(
        source="critic",
        path=should_continue,
        path_map={
            "end":      "section_summarizer",
            "continue": "summarizer",
        },
    )
 
    graph = workflow.compile()
    logger.info("ScholarGraph compiled successfully.")
    return graph
 
# COMPILED INSTANCE 
 
research_graph = build_graph()
 
 
def run_pipeline(
    pdf_path:        str,
    user_query:      str = "",
    expertise_level: str = "Intermediate",
) -> dict:
    """Convenience wrapper —> run the full pipeline."""
    initial_state: AgentState = {
        "pdf_path":              pdf_path,
        "user_query":            user_query,
        "expertise_level":       expertise_level,
        "raw_text":              None,
        "retriever":             None,
        "context":               [],
        "arxiv_papers":          [],
        "summary":               "",
        "critic_feedback":       "",
        "is_hallucination_free": False,
        "iteration_count":       0,
        "section_summaries":     None,
        "messages":              [],
    }
    return research_graph.invoke(initial_state)