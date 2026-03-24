from typing import Annotated, Optional, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Shared state for the ScholarGraph multi-agent pipeline.
 
    Flow:
        Ingestion → Planner → Researcher → Summarizer → Critic
                                                ↑____________|
                                           (loop on hallucination)
                                                ↓ (CLEAR)
                                       SectionSummarizer → END
    """
 
    # Initial Inputs
    pdf_path: str
    """Absolute or relative path to the uploaded PDF file."""
 
    user_query: str
    """The user's natural-language question or instruction."""
 
    expertise_level: str
    """
    Reader expertise level. Must be one of:
        - "Beginner"      → plain language, analogies, definitions
        - "Intermediate"  → balanced technical + accessible
        - "Expert"        → full technical depth, citations, model names
    """
 
    # RAG & Retrieval
    raw_text: Optional[str]
    """Full Markdown text extracted from the PDF by the Ingestion node."""
 
    retriever: Optional[Any]
    """
    FAISS retriever object built from the embedded PDF chunks.
    Typed as Any because LangGraph state must be serialisable in type hints,
    but the actual object is a LangChain VectorStoreRetriever.
    Passed in from app.py to avoid rebuilding it twice.
    """
 
    context: list[str]
    """
    Top-k text chunks retrieved from the FAISS vector store.
    Each item is a chunk of the original paper most relevant to the query.
    """
 
    arxiv_papers: list[str]
    """
    Titles or abstracts of related papers fetched from the ArXiv API.
    Used by the RAG agent to enrich Expert-level summaries.
    """
    # Citations 
    citations: Optional[list]
    """Extracted references list. Each item: {"number": "1", "text": "..."}"""
    # Agent Trace
    agent_trace: Optional[list]
    """
    List of step dicts recording what each agent did.
    Each: {"step": int, "agent": str, "summary": str, "llm": str}
    """
     # Visual Descriptions
    visual_descriptions: Optional[list]
    """
    List of dicts, one per extracted image/chart/table:
    {"page", "caption", "description", "base64_data", "media_type",
     "width", "height"}
    """

    # Summary & Review 
    summary: str
    """
    The current generated summary. May be rewritten multiple times
    if the Critic agent flags hallucinations or quality issues.
    """
 
    critic_feedback: str
    """
    Feedback from the Critic agent after cross-checking the summary
    against the original text. Empty string means the summary passed.
    """
 
    is_hallucination_free: bool
    """
    Flag set by the Critic agent.
        True  → summary passed review, graph moves to End node.
        False → summary failed, graph loops back to Synthesis agent.
    """
 
    iteration_count: int
    """
    Tracks how many Synthesis → Critic loops have occurred.
    The graph will force-exit after 3 iterations to prevent infinite loops.
    This is a critical safety guard for production systems.
    """
    
    # Detailed Section Summaries 
    section_summaries: Optional[dict]
    # Conversation History 
    messages: Annotated[list, add_messages]
    """
    Full conversation history between the user and the assistant.
    Uses LangChain's add_messages reducer so new messages are
    automatically appended rather than overwriting the list.
    """
    # Evaluation (Multi-dimensional)
    eval_accuracy:      Optional[int]
    """LLM-judged accuracy score 1-5: are claims supported by source?"""

    eval_completeness:  Optional[int]
    """LLM-judged completeness 1-5: are key points covered?"""

    eval_clarity:       Optional[int]
    """LLM-judged clarity 1-5: is it appropriate for the expertise level?"""

    eval_justifications: Optional[dict]
    """Dict of dimension → one-line justification from the Critic."""

    eval_readability_score:  Optional[float]
    """Flesch Reading Ease score (0-100). Higher = easier to read."""

    eval_readability_grade:  Optional[float]
    """Flesch-Kincaid Grade Level. Corresponds to US school grade."""

    eval_overall: Optional[float]
    """Average of accuracy + completeness + clarity (1-5)."""