import os
import logging
import time
import re
import base64 as b64_module
from dotenv import load_dotenv
from google import genai as genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.state import AgentState
from src.tools import research_paper, arxiv_search, extract_citations

load_dotenv()
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
 
GEMINI_MODEL   = "gemini-2.5-flash-lite"
GROQ_MODEL     = "llama-3.3-70b-versatile"
PRIMARY_MODEL  = "groq" 
MAX_ITERATIONS = 3
REQUEST_DELAY  = 1
 
# Session-level Gemini quota flag
_gemini_quota_exhausted: bool = False
 
# Module-level trace accumulator —> reset per pipeline run via initial_state
_agent_trace: list = []

def _record_trace(step: int, agent: str, summary: str, llm: str = ""):
    """Record one agent step to the module-level trace list."""
    _agent_trace.append({
        "step":    step,
        "agent":   agent,
        "summary": summary,
        "llm":     llm,
    })
    logger.info(f"Trace [{step}] {agent}: {summary}")

# LLM HELPERS

def _get_gemini_llm(temperature: float = 0.3):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set in .env")
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=temperature,
    )
 
 
def _get_groq_llm(temperature: float = 0.3):
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError("Run: pip install langchain-groq")
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env")
    return ChatGroq(
        model=GROQ_MODEL,
        groq_api_key=groq_key,
        temperature=temperature,
    )
 
 
def _safe_invoke(
    messages: list,
    temperature: float = 0.3,
    agent_name: str = "Agent",
) -> str:
    """
    Invoke LLM with session-level Gemini quota tracking.
    On first 429, instantly switches to Groq for all subsequent calls.
    """
    global _gemini_quota_exhausted
 
    time.sleep(REQUEST_DELAY)
 
    if PRIMARY_MODEL == "gemini" and not _gemini_quota_exhausted:
        try:
            llm      = _get_gemini_llm(temperature)
            response = llm.invoke(messages)
            logger.info(f"{agent_name}: Gemini succeeded.")
            return response.content.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                _gemini_quota_exhausted = True
                logger.warning(f"{agent_name}: Gemini exhausted —> switching to Groq.")
            else:
                logger.error(f"{agent_name}: Gemini error: {e}")
    else:
        logger.info(f"{agent_name}: Groq (Gemini exhausted this session).")
 
    try:
        groq_llm = _get_groq_llm(temperature)
        response = groq_llm.invoke(messages)
        logger.info(f"{agent_name}: Groq succeeded.")
        return response.content.strip()
    except Exception as e:
        logger.error(f"{agent_name}: Groq failed: {e}")
        return ""
 
 

# AGENT 1 —> PLANNER
 
def planner_agent(state: AgentState) -> dict:
    """Refines query. Skips LLM if user already provided one."""
    logger.info("Planner agent started.")
 
    raw_text   = state.get("raw_text", "")
    expertise  = state.get("expertise_level", "Intermediate")
    user_query = state.get("user_query", "")
 
    if user_query and len(user_query.strip()) > 10:
        logger.info("Planner: user query provided —> skipping LLM.")
        return {"user_query": user_query.strip(), "expertise_level": expertise}
 
    if not raw_text:
        return {
            "user_query": "Summarise the key contributions of this paper.",
            "expertise_level": expertise,
        }
 
    system_prompt = (
        "You are a research planning assistant. "
        "Read the paper excerpt and generate a focused 1-sentence search query "
        "capturing its core contribution. Respond with ONLY the query."
    )
    human_prompt = (
        f"Paper excerpt:\n{raw_text[:2000]}\n\n"
        f"Reader level: {expertise}\nGenerate a focused search query."
    )
 
    refined_query = _safe_invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)],
        temperature=0.1, agent_name="Planner"
    )
 
    if not refined_query:
        refined_query = "Summarise the key contributions and methodology of this paper."
 
    logger.info(f"Planner query: {refined_query[:100]}")
    _record_trace(1, "Planner", f"Query: '{refined_query[:80]}'" if 'refined_query' in dir() else "Used user query",
    "Groq" if _gemini_quota_exhausted else "Gemini")
    return {"user_query": refined_query, "expertise_level": expertise}
 
 
# AGENT 2 —> RESEARCHER
 
def researcher_agent(state: AgentState) -> dict:
    """RAG retrieval via FAISS. No LLM call —> zero quota used."""
    logger.info("Researcher agent started.")
 
    query     = state.get("user_query", "")
    expertise = state.get("expertise_level", "Intermediate")
    retriever = state.get("retriever")
 
    if not query:
        return {"context": [], "arxiv_papers": []}
 
    context_chunks = []
    if retriever is not None:
        rag_context = research_paper(query, retriever)
        if rag_context:
            context_chunks = [rag_context]
        logger.info(f"RAG: {len(context_chunks)} block(s).")
    else:
        logger.warning("Researcher: retriever is None.")
 
    arxiv_results = []
    if expertise == "Expert":
        arxiv_results = arxiv_search(query)
    

    citations = extract_citations(state.get("raw_text", ""))
    logger.info(f"Citations extracted: {len(citations)}")
    _record_trace(2, "Researcher",
        f"Retrieved {len(context_chunks)} RAG blocks, "
        f"{len(arxiv_results)} ArXiv papers, "
        f"{len(citations) if 'citations' in dir() else 0} citations",
        "FAISS (local)")
    return {"context": context_chunks, "arxiv_papers": arxiv_results, "citations": citations }
 
# AGENT 3 —> SUMMARIZER
 
_SUMMARIZER_PROMPTS = {
"Beginner": (
    "You are a friendly science communicator for a high-school student.\n"
    "Rules: Use simple language, analogies, and define jargon in [brackets].\n"
    "Structure your response exactly with these colored headers:\n\n"

    "### :red[Problem Statement]\n"
    "Explain: What problem does it solve?\n\n"
    "### :blue[Core Mechanism]\n"
    "Explain: How does it work?\n\n"
    "### :green[Key Findings]\n"
    "Explain: What did they find?\n\n"
    "### :violet[Global Impact]\n"
    "Explain: Why does it matter?"
    ),

    "Intermediate": (
        "You are a research assistant for a university student.\n"
        "Rules: balance accuracy with accessibility, explain key terms.\n"
        "Structure: Problem → Approach → Key Findings → Significance."
    ),
    "Expert": (
        "You are a peer reviewer for a senior researcher.\n"
        "Rules: precise technical language, full domain knowledge assumed.\n"
        "Focus: contributions, methodology, metrics, limitations.\n"
        "Structure: Contribution → Methodology → Results → Limitations."
    ),
}
 
 
def summarizer_agent(state: AgentState) -> dict:
    """Generates adaptive overall summary."""
    logger.info("Summarizer agent started.")
 
    context         = state.get("context", [])
    arxiv_papers    = state.get("arxiv_papers", [])
    expertise       = state.get("expertise_level", "Intermediate")
    critic_feedback = state.get("critic_feedback", "")
    iteration       = state.get("iteration_count", 0)
    raw_text        = state.get("raw_text", "")
 
    if context:
        context_text = "\n\n---\n\n".join(context)
    elif raw_text:
        logger.warning("Summarizer: using raw_text fallback.")
        context_text = raw_text[:4000]
    else:
        context_text = "No context available."
 
    context_text = context_text[:6000]
 
    if arxiv_papers and expertise == "Expert":
        context_text += "\n\n=== Related Papers ===\n" + "\n\n".join(arxiv_papers)
 
    system_prompt = _SUMMARIZER_PROMPTS.get(expertise, _SUMMARIZER_PROMPTS["Intermediate"])
 
    revision_note = ""
    if critic_feedback and critic_feedback.strip().upper() != "CLEAR":
        revision_note = f"\n\nFix these issues from your previous attempt:\n{critic_feedback}"
 
    human_prompt = (
        f"Write a summary for a {expertise}-level reader.{revision_note}\n\n"
        f"=== SOURCE CONTEXT ===\n{context_text}"
    )
 
    summary = _safe_invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)],
        temperature=0.4, agent_name="Summarizer"
    )
 
    if not summary:
        summary = (
            "⚠️ Summary generation failed.\n"
            "Check GROQ_API_KEY in .env or get one free at console.groq.com"
        )
 
    logger.info(f"Summarizer: {len(summary)} chars.")
    _record_trace(iteration + 1 + 2, "Summarizer",
        f"{len(summary)} chars generated (iteration {iteration + 1})",
        "Groq" if _gemini_quota_exhausted else "Gemini")
    return {"summary": summary, "iteration_count": iteration + 1}
 


def _compute_readability(text: str) -> dict:
    """
    Computes Flesch Reading Ease and Flesch-Kincaid Grade Level locally.
    No LLM call. Uses textstat library if available, falls back to
    a simple approximation if not installed.

    Returns:
        Dict with eval_readability_score and eval_readability_grade.
    """
    if not text or len(text.split()) < 10:
        return {"eval_readability_score": None, "eval_readability_grade": None}

    try:
        import textstat
        score = round(textstat.flesch_reading_ease(text), 1)
        grade = round(textstat.flesch_kincaid_grade(text), 1)
        logger.info(f"Readability: Flesch={score}, FK Grade={grade}")
        return {
            "eval_readability_score": score,
            "eval_readability_grade": grade,
        }
    except ImportError:
        # Simple approximation without textstat
        words     = text.split()
        sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
        syllables = sum(_count_syllables(w) for w in words)
        asl       = len(words) / sentences          # avg sentence length
        asw       = syllables / max(1, len(words))  # avg syllables per word
        flesch    = round(206.835 - 1.015 * asl - 84.6 * asw, 1)
        fk_grade  = round(0.39 * asl + 11.8 * asw - 15.59, 1)
        logger.info(f"Readability (approx): Flesch={flesch}, FK Grade={fk_grade}")
        return {
            "eval_readability_score": flesch,
            "eval_readability_grade": fk_grade,
        }


def _count_syllables(word: str) -> int:
    """Rough syllable counter for readability fallback."""
    word    = word.lower().strip(".,!?;:'\"")
    vowels  = "aeiouy"
    count   = 0
    prev_v  = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_v:
            count += 1
        prev_v = is_v
    return max(1, count)


def _parse_critic_response(response: str) -> dict:
    """
    Parses the structured critic response into a clean dict.

    Expected format:
        VERDICT: CLEAR or ISSUES
        ISSUES: <text or 'None'>
        ACCURACY: <1-5> | <justification>
        COMPLETENESS: <1-5> | <justification>
        CLARITY: <1-5> | <justification>

    Falls back gracefully if format is not followed exactly.
    """
    result = {
        "verdict":        "CLEAR",
        "issues":         "CLEAR",
        "accuracy":       3,
        "completeness":   3,
        "clarity":        3,
        "justifications": {},
    }

    lines = response.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        upper = line.upper()

        if upper.startswith("VERDICT:"):
            val = line.split(":", 1)[1].strip().upper()
            result["verdict"] = "CLEAR" if "CLEAR" in val else "ISSUES"

        elif upper.startswith("ISSUES:"):
            val = line.split(":", 1)[1].strip()
            result["issues"] = val if val.lower() not in ("none", "n/a", "") else "CLEAR"

        elif upper.startswith("ACCURACY:"):
            score, just = _parse_score_line(line)
            result["accuracy"] = score
            result["justifications"]["Accuracy"] = just

        elif upper.startswith("COMPLETENESS:"):
            score, just = _parse_score_line(line)
            result["completeness"] = score
            result["justifications"]["Completeness"] = just

        elif upper.startswith("CLARITY:"):
            score, just = _parse_score_line(line)
            result["clarity"] = score
            result["justifications"]["Clarity"] = just

    return result


def _parse_score_line(line: str) -> tuple[int, str]:
    """
    Parses a line like 'ACCURACY: 4 | Claims are well supported.'
    Returns (score_int, justification_str).
    Falls back to score=3 if parsing fails.
    """
    try:
        after_colon = line.split(":", 1)[1].strip()
        if "|" in after_colon:
            score_part, justification = after_colon.split("|", 1)
        else:
            score_part     = after_colon
            justification  = ""

        # Extract first digit found
        digits = re.findall(r"\d", score_part)
        score  = int(digits[0]) if digits else 3
        score  = max(1, min(5, score))   # clamp to 1-5
        return score, justification.strip()
    except Exception:
        return 3, ""
 
# AGENT 4 —> CRITIC
 
def critic_agent(state: AgentState) -> dict:
    """
    Fact-checks summary AND returns structured evaluation scores.

    Evaluation dimensions (each 1-5):
        Accuracy     —> are claims supported by the source context?
        Completeness —> are the key points of the paper covered?
        Clarity      —> is the language appropriate for the expertise level?

    Also computes local readability metrics (no LLM call):
        Flesch Reading Ease
        Flesch-Kincaid Grade Level

    Reads:  summary, context, raw_text, iteration_count, expertise_level
    Writes: critic_feedback, is_hallucination_free,
            eval_accuracy, eval_completeness, eval_clarity,
            eval_justifications, eval_readability_score,
            eval_readability_grade, eval_overall
    """
    logger.info("Critic agent started.")

    summary   = state.get("summary", "")
    context   = state.get("context", [])
    raw_text  = state.get("raw_text", "")
    iteration = state.get("iteration_count", 0)
    expertise = state.get("expertise_level", "Intermediate")

    # Default values if summary is empty or LLM fails —> ensures Critic always returns a complete eval dict
    default_eval = {
        "critic_feedback":       "CLEAR",
        "is_hallucination_free": True,
        "eval_accuracy":         None,
        "eval_completeness":     None,
        "eval_clarity":          None,
        "eval_justifications":   {},
        "eval_readability_score": None,
        "eval_readability_grade": None,
        "eval_overall":          None,
    }

    # Hard fail conditions —> no context to check against, or summary contains clear error message from LLM
    if iteration >= MAX_ITERATIONS:
        logger.warning("Max iterations. Forcing CLEAR.")
        readability = _compute_readability(summary)
        return {**default_eval, **readability}

    if not summary or "⚠️" in summary:
        return {**default_eval,
                "critic_feedback": "Summary failed.",
                "is_hallucination_free": False}

    # Use context if available, otherwise fall back to raw_text for fact-checking. If neither is available, skip straight to readability evaluation.
    if context:
        context_text = "\n\n---\n\n".join(context)[:3000]
    elif raw_text:
        context_text = raw_text[:3000]
    else:
        readability = _compute_readability(summary)
        return {**default_eval, **readability}

    # Parse the critic's structured response
    system_prompt = (
        "You are a rigorous scientific fact-checker and writing evaluator.\n\n"
        "Your job is to:\n"
        "1. Check if the summary is factually supported by the source context.\n"
        "2. Score the summary on three dimensions (each 1-5).\n\n"
        "Respond in this EXACT format with no extra text:\n\n"
        "VERDICT: CLEAR or ISSUES\n"
        "ISSUES: <list specific unsupported claims, or 'None' if CLEAR>\n"
        "ACCURACY: <1-5> | <one-line justification>\n"
        "COMPLETENESS: <1-5> | <one-line justification>\n"
        "CLARITY: <1-5> | <one-line justification>\n\n"
        "Scoring guide:\n"
        f"- Accuracy: are all claims directly supported by the source?\n"
        f"- Completeness: are the key contributions and findings covered?\n"
        f"- Clarity: is the language appropriate for a {expertise}-level reader?\n"
        "5=Excellent, 4=Good, 3=Adequate, 2=Weak, 1=Poor"
    )
    human_prompt = (
        f"=== SOURCE CONTEXT ===\n{context_text}\n\n"
        f"=== SUMMARY TO EVALUATE ===\n{summary}\n\n"
        "Evaluate following the exact format above."
    )

    raw_response = _safe_invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)],
        temperature=0.0,
        agent_name="Critic"
    )

    if not raw_response:
        logger.warning("Critic: LLM returned empty. Auto-passing.")
        readability = _compute_readability(summary)
        return {**default_eval, **readability}

    # Parse the critic's structured response and compute readability metrics
    parsed = _parse_critic_response(raw_response)
    readability = _compute_readability(summary)

    is_clear    = parsed.get("verdict", "CLEAR").upper() == "CLEAR"
    feedback    = parsed.get("issues", "CLEAR") if not is_clear else "CLEAR"

    accuracy     = parsed.get("accuracy", 3)
    completeness = parsed.get("completeness", 3)
    clarity      = parsed.get("clarity", 3)
    overall      = round((accuracy + completeness + clarity) / 3, 1)

    logger.info(
        f"Critic: {'CLEAR' if is_clear else 'Issues'} | "
        f"A:{accuracy} C:{completeness} Cl:{clarity} → {overall}/5"
    )
    _record_trace(0, "Critic",
        f"{'CLEAR' if is_clear else 'Issues found'} | "
        f"A:{accuracy} C:{completeness} Cl:{clarity} → {overall}/5",
        "Groq" if _gemini_quota_exhausted else "Gemini")
    return {
        "critic_feedback":        feedback,
        "is_hallucination_free":  is_clear,
        "eval_accuracy":          accuracy,
        "eval_completeness":      completeness,
        "eval_clarity":           clarity,
        "eval_justifications":    parsed.get("justifications", {}),
        "eval_readability_score": readability.get("eval_readability_score"),
        "eval_readability_grade": readability.get("eval_readability_grade"),
        "eval_overall":           overall,
    }
 
 
# AGENT 5 —> SECTION SUMMARIZER
 
def _is_heading_line(line: str) -> bool:
    """
    Returns True if a line looks like a section heading.

    Handles ALL common academic paper formats:
    - Markdown:       ## Introduction
    - Bold markdown:  **Introduction**
    - ALL CAPS:       INTRODUCTION
    - Roman numeral:  IV. RESULTS
    - Numbered:       3. Methods  /  3) Methods
    - Plain short:    Introduction  (short line, title-cased, no period)
    """
    s = line.strip()
    if len(s) < 3 or len(s) > 120:
        return False

    # Common patterns for section headings in academic papers
    if re.match(r"^#{1,4}\s+\S", s):
        return True
    # Bold markdown headings (e.g., **Introduction**)
    if re.match(r"^\*\*[^*]{3,80}\*\*\s*$", s):
        return True
    # ALL CAPS headings (but not if it's just a number or short word)
    if s.isupper() and len(s) > 3 and not re.match(r"^\d+$", s):
        return True
    # Roman numeral headings (e.g., IV. RESULTS)
    if re.match(r"^[IVXLC]+\.\s+[A-Z]", s):
        return True
    # Numbered headings (e.g., 3. Methods or 3) Methods)
    if re.match(r"^\d{1,2}[\.\)]\s+[A-Z]", s):
        return True
    # Plain title-cased lines (e.g., Introduction) — only if short and not ending with punctuation
    if re.match(r"^\d{1,2}\s+[A-Z][a-z]", s):
        return True
    # Additional heuristic: if it's short (2-6 words), title-cased, and doesn't end with a period, it might be a heading
    words = s.split()
    if (2 <= len(words) <= 6
            and not s.endswith(".")
            and not s.endswith(",")
            and not s.endswith(":")
            and s[0].isupper()
            and not any(c.isdigit() for c in s)):
        
        stop_words = {"the", "a", "an", "and", "or", "but", "in",
                      "on", "at", "to", "for", "of", "with", "is",
                      "are", "was", "were", "be", "been", "this",
                      "that", "these", "those", "we", "our", "their"}
        lower_words = {w.lower() for w in words}
        # Only treat as heading if few stop words (real headings are terse)
        if len(lower_words & stop_words) <= 1:
            return True

    return False


def _clean_heading_text(line: str) -> str:
    """Strip Markdown symbols from a heading to get plain text."""
    s = line.strip()
    s = re.sub(r"^#{1,4}\s*", "", s)           # remove leading #
    s = re.sub(r"^\*\*(.*)\*\*$", r"\1", s)    # unwrap **bold**
    s = re.sub(r"^\d+[\.\)]\s*", "", s)         # remove leading numbers
    s = re.sub(r"^[IVXLC]+\.\s*", "", s)        # remove Roman numerals
    return s.strip()


# Known section name patterns for priority matching
_KNOWN_SECTION_PATTERNS = {
    "Abstract":     r"(?i)\b(abstract)\b",
    "Introduction": r"(?i)\b(introduction|background|overview|motivation)\b",
    "Related Work": r"(?i)\b(related\s*work|literature\s*review|prior\s*work|related\s*literature)\b",
    "Methodology":  r"(?i)\b(method|methodology|approach|proposed|framework|architecture|system\s*design|implementation|model)\b",
    "Experiments":  r"(?i)\b(experiment|experimental\s*setup|experimental\s*results|empirical)\b",
    "Results":      r"(?i)\b(result|evaluation|performance|benchmark|ablation)\b",
    "Discussion":   r"(?i)\b(discussion|analysis|findings|limitation)\b",
    "Conclusion":   r"(?i)\b(conclusion|future\s*work|summary|closing)\b",
}


def _detect_sections(raw_text: str) -> dict:
    """
    Two-pass section detector that handles both standard and
    non-standard paper formats.

    Pass 1 — Match against 8 known academic section patterns.
    Pass 2 — If fewer than 2 known sections found, extract ANY heading
             that passes _is_heading_line() regardless of its name.
             This handles custom section names like:
             'Proposed System', 'Experimental Setup', 'Dataset Description',
             'Future Directions', 'Ablation Study', etc.

    Args:
        raw_text: Full Markdown text from ingest_pdf().

    Returns:
        OrderedDict: {display_name: section_text}
    """
    lines      = raw_text.split("\n")
    heading_map = []   

    # Pass 1: Look for known section patterns first —> preserves standard structure when available
    known_found = set()

    for i, line in enumerate(lines):
        if not _is_heading_line(line):
            continue
        clean = _clean_heading_text(line)
        for section_name, pattern in _KNOWN_SECTION_PATTERNS.items():
            if re.search(pattern, clean) and section_name not in known_found:
                heading_map.append((i, section_name))
                known_found.add(section_name)
                break

    logger.info(f"Section detector Pass 1: found {len(known_found)} known sections: {list(known_found)}")

    # Pass 2: If fewer than 2 known sections found, extract ANY heading line as a section —> ensures we get some structure even from non-standard papers
    if len(known_found) < 2:
        logger.info("Section detector: switching to Pass 2 (generic heading extraction).")
        heading_map = []
        seen_norms  = set()

        for i, line in enumerate(lines):
            if not _is_heading_line(line):
                continue
            clean = _clean_heading_text(line)

            # Skip clearly non-section lines
            if len(clean) < 3:
                continue
            if re.match(r"^\d+$", clean):           
                continue
            if re.match(r"(?i)^(fig|figure|table)\s*\d", clean): 
                continue
            if re.match(r"(?i)^references?\s*$", clean):  
                continue

            # Normalise for deduplication
            norm = re.sub(r"^\d+[\.\s]*", "", clean).strip().lower()
            if norm and norm not in seen_norms:
                seen_norms.add(norm)
                heading_map.append((i, clean))

    if not heading_map:
        logger.warning("Section detector: no headings found in document.")
        return {}

    heading_map.sort(key=lambda x: x[0])

    # Build the final section text mapping based on detected headings
    detected = {}
    for idx, (line_idx, display_name) in enumerate(heading_map):
        end_line = (
            heading_map[idx + 1][0]
            if idx + 1 < len(heading_map)
            else len(lines)
        )
        section_text = "\n".join(lines[line_idx:end_line]).strip()

        # Skip sections with very little content (false positives)
        if len(section_text) < 150:
            continue

        detected[display_name] = section_text

    logger.info(f"Section detector final: {len(detected)} sections — {list(detected.keys())}")
    return detected
 
# Critical sections we ALWAYS want —> regardless of paper formatting
_CRITICAL_SECTIONS = {
    "Abstract": (
        "What is this paper fundamentally about? "
        "What is its core contribution or claim?"
    ),
    "Problem Statement": (
        "What specific problem or gap does this paper address? "
        "Why is this problem important or unsolved?"
    ),
    "Approach / Methodology": (
        "How do the authors solve the problem? "
        "What methods, models, algorithms, or frameworks do they propose or use?"
    ),
    "Results & Findings": (
        "What did the experiments or analysis show? "
        "What are the key numbers, benchmarks, or qualitative findings?"
    ),
    "Conclusion & Impact": (
        "What are the main takeaways? "
        "What future work is suggested? What is the broader impact?"
    ),
}


def section_summarizer_agent(state: AgentState) -> dict:
    """
    Generates summaries for paper sections using a two-layer strategy:

    Layer 1 — Detected sections (structure-aware):
        Runs the heading detector on raw_text. Any section found explicitly
        gets a summary using its actual content. This preserves the paper's
        own organisation.

    Layer 2 — Guaranteed critical sections (content-aware):
        For each of 5 critical sections (Abstract, Problem, Approach,
        Results, Conclusion), if not already covered by Layer 1, asks the
        LLM to extract and summarise it from the full raw text.
        This ensures every paper — regardless of formatting — produces
        at least 5 meaningful summaries.

    The final output merges both layers:
        detected sections first → critical sections below.

    Reads:  raw_text, expertise_level
    Writes: section_summaries (dict: section_name → summary_string)
    """
    logger.info("Section summarizer agent started.")

    raw_text  = state.get("raw_text", "")
    expertise = state.get("expertise_level", "Intermediate")

    if not raw_text:
        logger.warning("Section summarizer: raw_text is empty.")
        return {"section_summaries": {}}

    section_summaries = {}

    # Layer 1: Detect and summarise explicitly marked sections based on headings in the raw text
    detected = _detect_sections(raw_text)

    if detected:
        logger.info(f"Layer 1: summarising {len(detected)} detected sections.")
        for section_name, section_text in detected.items():
            preview = section_text[:3500]
            system_prompt = (
                f"You are summarising the '{section_name}' section of a "
                f"research paper for a {expertise}-level reader.\n"
                "Write a focused 3-5 sentence summary of ONLY this section.\n"
                "Be specific — include key methods, numbers, or findings.\n"
                "Do not start with 'This section discusses...'"
            )
            human_prompt = (
                f"=== {section_name.upper()} ===\n{preview}\n\n"
                f"Write a concise {expertise}-level summary."
            )
            result = _safe_invoke(
                [SystemMessage(content=system_prompt),
                 HumanMessage(content=human_prompt)],
                temperature=0.3,
                agent_name=f"SectionSummarizer[{section_name}]"
            )
            if result:
                section_summaries[section_name] = result
                logger.info(f"  ✓ {section_name}: {len(result)} chars")
    else:
        logger.warning("Layer 1: no sections detected.")

    # Layer 2: For critical sections, if not already covered by detected headings, extract from full text using LLM —> ensures we get key info even from non-standard papers
    covered_text = " ".join(section_summaries.keys()).lower()

    coverage_keywords = {
        "Abstract":              ["abstract"],
        "Problem Statement":     ["problem", "introduction", "motivation", "background"],
        "Approach / Methodology":["method", "approach", "framework", "proposed",
                                  "architecture", "model", "system", "implementation"],
        "Results & Findings":    ["result", "evaluation", "experiment", "finding",
                                  "performance", "benchmark"],
        "Conclusion & Impact":   ["conclusion", "summary", "future", "impact"],
    }

    # To save tokens, we only pass a preview of the raw text to the LLM for critical section extraction. The LLM can still find key info if it's in the first 6000 chars, which is common for abstracts, introductions, and sometimes conclusions. If the paper is very long and the critical info is buried deep, it may be missed — but this is a tradeoff to stay within token limits and avoid excessive costs.
    raw_preview = raw_text[:6000]

    for critical_name, question in _CRITICAL_SECTIONS.items():
        # Check if this critical area is already covered
        keywords = coverage_keywords.get(critical_name, [])
        already_covered = any(kw in covered_text for kw in keywords)

        if already_covered:
            logger.info(f"  Layer 2: '{critical_name}' already covered — skipping.")
            continue

        logger.info(f"  Layer 2: extracting '{critical_name}' from full text.")

        system_prompt = (
            f"You are a research assistant analysing a scientific paper "
            f"for a {expertise}-level reader.\n"
            f"Your task: {question}\n\n"
            "Rules:\n"
            "- Answer in 3-5 sentences.\n"
            "- Base your answer ONLY on the provided paper text.\n"
            "- If this information is not present in the text, "
            "  respond with exactly: NOT FOUND\n"
            "- Do not invent or assume information."
        )
        human_prompt = (
            f"=== PAPER TEXT ===\n{raw_preview}\n\n"
            f"Based on this text, answer: {question}"
        )

        result = _safe_invoke(
            [SystemMessage(content=system_prompt),
             HumanMessage(content=human_prompt)],
            temperature=0.2,
            agent_name=f"CriticalSection[{critical_name}]"
        )

        if result and result.strip().upper() != "NOT FOUND" and len(result) > 50:
            section_summaries[critical_name] = result
            logger.info(f"  ✓ {critical_name}: {len(result)} chars (extracted)")
        else:
            logger.info(f"  - {critical_name}: not found in text.")

    logger.info(
        f"Section summarizer complete: {len(section_summaries)} sections "
        f"({len(detected)} detected + "
        f"{len(section_summaries) - len(detected)} critical extractions)"
    )
    
    _record_trace(99, "SectionSummarizer",
        f"{len(section_summaries)} sections "
        f"({len(detected)} detected + "
        f"{len(section_summaries)-len(detected)} extracted)",
        "Groq" if _gemini_quota_exhausted else "Gemini")
    
    return {"section_summaries": section_summaries, "agent_trace": list(_agent_trace)}


# CHATBOT —> Conversational RAG

def chat_with_paper(
    user_message:   str,
    retriever,
    chat_history:   list[dict],
    expertise_level: str = "Intermediate",
    provider:       str  = "groq",
    api_key:        str  = "",
) -> str:
    """
    Answer a user question about the paper using conversational RAG.

    Strategy:
        1. Retrieve top-4 relevant chunks from FAISS for the user's question
        2. Build a prompt with retrieved context + last 6 turns of history
        3. Call the selected LLM provider
        4. Return the response

    Args:
        user_message   : The user's question.
        retriever      : FAISS retriever built from the ingested paper.
        chat_history   : List of {"role": "user"/"assistant", "content": "..."}.
        expertise_level: Adjusts response style (Beginner/Intermediate/Expert).
        provider       : "groq", "gemini", or "cohere".
        api_key        : User-provided API key for the chosen provider.

    Returns:
        Assistant response string. Returns error message on failure.
    """
    logger.info(f"Chat: '{user_message[:80]}' | provider={provider}")

    # Step 1: Retrieve relevant context from the paper using the retriever. If retrieval fails, we proceed with an empty context but still allow the chatbot to answer based on conversation history.
    context = ""
    if retriever:
        try:
            context = research_paper(user_message, retriever)
        except Exception as e:
            logger.warning(f"Chat RAG retrieval failed: {e}")

    # Step 2: Build system prompt based on expertise level
    level_instructions = {
        "Beginner":     "Use simple language and analogies. Avoid jargon.",
        "Intermediate": "Balance technical accuracy with accessibility.",
        "Expert":       "Use precise technical language. Assume full domain knowledge.",
    }
    level_note = level_instructions.get(expertise_level, level_instructions["Intermediate"])

    system_prompt = (
        "You are a helpful research assistant answering questions about "
        "an attached scientific paper.\n\n"
        f"Reader level: {expertise_level}. {level_note}\n\n"
        "Rules:\n"
        "- Base your answer ONLY on the provided paper context.\n"
        "- If the answer is not in the context, say clearly: "
        "  'This information is not available in the provided paper.'\n"
        "- Be concise but complete.\n"
        "- For follow-up questions, use the conversation history for context."
    )

    # Step 3: Build the message list for the LLM, including system prompt, conversation history, and current question with retrieved context. We include up to the last 6 turns of history to give the LLM enough context without exceeding token limits. The retrieved context is included directly in the user message to ensure it's considered in the response.
    messages = [SystemMessage(content=system_prompt)]

    # Include last 6 turns of conversation history for context (both user and assistant). This helps the LLM maintain continuity in the conversation and reference previous questions and answers.
    for turn in chat_history[-6:]:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            
            messages.append(AIMessage(content=turn["content"]))
    # Add the current user question, along with the retrieved context from the paper. If no context was retrieved, we still include the user's question but note that no specific context is available. This allows the chatbot to attempt an answer based on conversation history or to clearly state that the information is not in the paper.
    if context:
        user_content = (
            f"=== RELEVANT PAPER CONTEXT ===\n{context}\n\n"
            f"=== MY QUESTION ===\n{user_message}"
        )
    else:
        user_content = (
            f"No specific context was retrieved for this question.\n\n"
            f"=== MY QUESTION ===\n{user_message}"
        )

    messages.append(HumanMessage(content=user_content))

    # Step 4: Call the selected LLM provider with the constructed messages. We wrap this in a try-except block to handle any potential errors gracefully. If the LLM call fails, we return a clear error message to the user instead of crashing the app.
    try:
        llm = _get_chat_llm(provider, api_key)
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Chat LLM failed: {e}")
        return (
            f"Chat failed with {provider.capitalize()}.\n\n"
            f"Error: {e}\n\n"
            f"Please check your API key in the sidebar."
        )


def _get_chat_llm(provider: str, api_key: str):
    """
    Returns an LLM instance for the chatbot based on provider choice.

    Supported:
        groq   → Llama 3.3 70B (free, fast, generous quota)
        gemini → Gemini 2.5 Flash Lite (free tier)
        cohere → Command R+ (free tier, strong at RAG tasks)

    Falls back to env key if no api_key provided in UI.
    """
    provider = provider.lower().strip()

    if provider == "groq":
        from langchain_groq import ChatGroq
        key = api_key.strip() or os.getenv("GROQ_API_KEY", "")
        if not key:
            raise ValueError(
                "Groq API key required.\n"
                "Get a free key at: console.groq.com"
            )
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=key,
            temperature=0.3,
        )

    elif provider == "gemini":
        key = api_key.strip() or os.getenv("GOOGLE_API_KEY", "")
        if not key:
            raise ValueError(
                "Gemini API key required.\n"
                "Get a free key at: aistudio.google.com"
            )
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=key,
            temperature=0.3,
        )

    elif provider == "cohere":
        try:
            from langchain_cohere import ChatCohere
        except ImportError:
            raise ImportError("Run: pip install langchain-cohere")
        key = api_key.strip() or os.getenv("COHERE_API_KEY", "")
        if not key:
            raise ValueError(
                "Cohere API key required.\n"
                "Get a free key at: cohere.com"
            )
        return ChatCohere(
            model="command-r-08-2024",
            cohere_api_key=key,
            temperature=0.3,
        )

    elif provider == "cerebras":
        try:
            from langchain_cerebras import ChatCerebras
        except ImportError:
            raise ImportError("Run: pip install langchain-cerebras")
        key = api_key.strip() or os.getenv("CEREBRAS_API_KEY", "")
        if not key:
            raise ValueError(
                "Cerebras API key required.\n"
                "Get a free key at: cloud.cerebras.ai"
            )
        return ChatCerebras(
            model="llama3.1-8b",
            cerebras_api_key=key,
            temperature=0.3,
        )

    else:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            "Supported: groq, gemini, cohere, cerebras"
        )


# VISION AGENT —> Describes images/charts/tables using a vision LLM

def describe_visuals(
    pdf_path:        str,
    expertise_level: str = "Intermediate",
    api_key:         str = "",
) -> list[dict]:
    """
    Three-layer visual analysis:
        Layer 1: Extract image bytes via PyMuPDF
        Layer 2: Find nearest caption + paper's own discussion of the figure
        Layer 3: Gemini Vision description (skipped gracefully if quota exhausted)

    If vision quota is exhausted after the first image, remaining images
    still return with their caption and paper discussion —> giving users
    meaningful content without requiring the LLM.
    """
    from src.tools import extract_images_from_pdf

    logger.info(f"Vision agent: extracting images from {pdf_path}")
    images = extract_images_from_pdf(pdf_path)

    if not images:
        logger.info("Vision agent: no images found.")
        return []

    logger.info(f"Vision agent: processing {len(images)} image(s).")

    level_instructions = {
        "Beginner":     (
            "Explain what this visual shows in simple everyday language. "
            "Use analogies. Avoid technical jargon."
        ),
        "Intermediate": (
            "Describe the key findings or structure shown. "
            "Mention axis labels, trends, or key values if visible."
        ),
        "Expert":       (
            "Provide a precise technical description. Include axis labels, "
            "metric values, trends, statistical patterns, or architectural "
            "components. Note any surprising or significant findings."
        ),
    }
    level_note = level_instructions.get(
        expertise_level, level_instructions["Intermediate"]
    )

    described      = []
    quota_exhausted = False   

    for img in images:
        caption    = img.get("caption", "")
        context    = img.get("context", "")
        b64_data   = img.get("base64_data", "")

       
        caption_note = f"This image has the caption: '{caption}'\n\n" if caption else ""
        context_note = (
            f"Surrounding text from paper: {context[:200]}\n\n"
            if context else ""
        )
        prompt = (
            f"{caption_note}{context_note}"
            f"Describe what this visual shows in 3-5 sentences. {level_note}\n"
            "Identify: (1) type of visual (bar chart/line graph/table/"
            "flow diagram/architecture diagram/scatter plot/etc.), "
            "(2) what it measures or represents, "
            "(3) the key insight or finding it communicates."
        )

        if not quota_exhausted and b64_data:
            raw_description = _call_vision_llm(
                prompt, b64_data, img["media_type"], 
                api_key=api_key,
            )
            if raw_description == "__QUOTA_EXHAUSTED__":
                quota_exhausted = True
                description = None   
            else:
                description = raw_description
        else:
            description = None

        if not description:
            if caption:
                description = (
                    f"[Vision description unavailable — Gemini quota exhausted]\n\n"
                    f"**From the paper:** {caption}"
                    + (f"\n\n**Context:** {context[:300]}" if context else "")
                )
            else:
                description = (
                    "[Vision description unavailable —> Gemini quota exhausted. "
                    "Image displayed above. Try again when quota resets at midnight UTC.]"
                )

        described.append({
            "page":        img["page"],
            "caption":     caption,
            "description": description,
            "base64_data": b64_data,
            "media_type":  img["media_type"],
            "width":       img["width"],
            "height":      img["height"],
            "has_vision":  description and "__QUOTA_EXHAUSTED__" not in description
                           and "Vision description unavailable" not in description,
        })

    vision_count = sum(1 for d in described if d.get("has_vision"))
    logger.info(
        f"Vision agent: {len(described)} visuals processed, "
        f"{vision_count} with AI descriptions."
    )
    return described

logger = logging.getLogger(__name__)

def _call_vision_llm(prompt: str, base64_data: str, media_type: str, api_key: str = "",) -> str:
    """
    Send image + prompt to Gemini Vision for description.
    Uses gemini-3.1-flash-lite-preview —> best free vision model available.

    Falls back gracefully on quota exhaustion with a helpful message
    that still shows the caption context.
    """
    time.sleep(1)

    # User provided key takes priority, then fall back to .env
    resolved_key = api_key.strip() or os.getenv("GOOGLE_API_KEY", "")
    if not resolved_key:
        return (
            "Vision description unavailable — no API key provided.\n"
            "Enter your Google API key in the sidebar to enable vision descriptions.\n"
            "Get a free key at: aistudio.google.com"
        )
    api_key = resolved_key

    try:
        from google import genai as google_genai
       
        client = google_genai.Client(api_key=api_key)

        image_part = types.Part.from_bytes(
            data=b64_module.b64decode(base64_data),
            mime_type=media_type,
        )

        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[image_part, prompt],
        )
        description = response.text.strip()
        logger.info(f"Vision LLM: {len(description)} chars generated.")
        return description

    except Exception as e:
        error_str = str(e)
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            logger.warning("Vision LLM: quota exhausted.")
            return "__QUOTA_EXHAUSTED__"
        else:
            logger.error(f"Vision LLM failed: {e}")
            return f"Vision description failed: {e}"
        

# DUAL LLM COMPARISON —> Generates summaries from two LLMs simultaneously

def generate_comparison_summaries(
    context:        str,
    raw_text:       str  = "",
    expertise_level: str = "Intermediate",
    llm_a_provider: str  = "groq",
    llm_b_provider: str  = "cerebras",
    llm_c_provider: str  = "gemini",
    llm_a_key:      str  = "",
    llm_b_key:      str  = "",
    llm_c_key:      str  = "",
    reference_summary: str = "",
) -> dict:
    """
    Three-way LLM comparison with automatic metric evaluation.

    Generates summaries from 3 LLMs and computes:
        - Readability (Flesch-Kincaid)
        - Word count + sentence count
        - Lexical diversity (unique/total words)
        - Similarity to main pipeline summary

    Args:
        context          : RAG-retrieved paper context.
        raw_text         : Full paper text (fallback if context empty).
        expertise_level  : Beginner / Intermediate / Expert.
        llm_a/b/c_provider: Provider names.
        llm_a/b/c_key    : Optional API keys (falls back to env).
        reference_summary: The main pipeline summary for similarity scoring.

    Returns:
        {
            "llm_a": {provider, model, summary, metrics},
            "llm_b": {provider, model, summary, metrics},
            "llm_c": {provider, model, summary, metrics},
            "winner": str,   # provider with best overall score
            "expertise_level": str,
        }
    """
    logger.info(
        f"3-way comparison: {llm_a_provider} vs {llm_b_provider} vs {llm_c_provider}"
    )

    source = context[:4000] if context else raw_text[:4000]
    system_prompt = _SUMMARIZER_PROMPTS.get(
        expertise_level, _SUMMARIZER_PROMPTS["Intermediate"]
    )
    human_prompt = (
        f"Write a summary for a {expertise_level}-level reader.\n\n"
        f"=== SOURCE CONTEXT ===\n{source}"
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    model_display_names = {
        "groq":     "Llama 3.3 70B · Groq",
        "cerebras": "Llama 3.1 8B · Cerebras",
        "gemini":   "Gemini 2.5 Flash Lite",
        "cohere":   "Command R · Cohere",
    }

    def _call(provider: str, api_key: str) -> dict:
        """Call one provider, return result with metrics."""
        try:
            llm      = _get_chat_llm(provider, api_key)
            response = llm.invoke(messages)
            summary  = response.content.strip()
            metrics  = _compute_summary_metrics(summary, reference_summary)
            logger.info(f"Comparison [{provider}]: {len(summary)} chars.")
            return {
                "provider":  provider,
                "model":     model_display_names.get(provider, provider),
                "summary":   summary,
                "metrics":   metrics,
                "error":     False,
            }
        except Exception as e:
            logger.error(f"Comparison [{provider}] failed: {e}")
            return {
                "provider":  provider,
                "model":     model_display_names.get(provider, provider),
                "summary":   f"❌ {provider.capitalize()} unavailable: {e}",
                "metrics":   _empty_metrics(),
                "error":     True,
            }

    result_a = _call(llm_a_provider, llm_a_key)
    time.sleep(1)
    result_b = _call(llm_b_provider, llm_b_key)
    time.sleep(1)
    result_c = _call(llm_c_provider, llm_c_key)
    winner = _find_winner([result_a, result_b, result_c])

    return {
        "llm_a":           result_a,
        "llm_b":           result_b,
        "llm_c":           result_c,
        "winner":          winner,
        "expertise_level": expertise_level,
    }


def _compute_summary_metrics(summary: str, reference: str = "") -> dict:
    """
    Compute local quality metrics for a summary.
    No LLM calls — all computed instantly from text statistics.

    Metrics:
        readability_score : Flesch Reading Ease (0-100)
        readability_grade : Flesch-Kincaid Grade Level
        word_count        : Total words
        sentence_count    : Total sentences
        lexical_diversity : Unique words / total words (0-1)
                            Higher = richer vocabulary
        similarity_score  : Cosine similarity to reference summary (0-1)
                            Higher = more consistent with main pipeline
        composite_score   : Weighted average of all metrics (0-10)
    """
    if not summary or "❌" in summary:
        return _empty_metrics()

    words     = summary.split()
    sentences = max(1, summary.count(".") + summary.count("!") + summary.count("?"))
    word_count     = len(words)
    sentence_count = sentences

    # Lexical diversity
    unique_words     = len(set(w.lower().strip(".,!?;:") for w in words))
    lexical_diversity = round(unique_words / max(1, word_count), 3)

    # Readability via textstat or fallback
    readability = _compute_readability(summary)
    flesch      = readability.get("eval_readability_score") or 50.0
    fk_grade    = readability.get("eval_readability_grade") or 10.0

    # Similarity to reference summary (if provided and valid)
    similarity = 0.0
    if reference and not reference.startswith("❌"):
        ref_words  = set(reference.lower().split())
        sum_words  = set(summary.lower().split())
        if ref_words or sum_words:
            intersection = ref_words & sum_words
            union        = ref_words | sum_words
            similarity   = round(len(intersection) / max(1, len(union)), 3)

    # Composite score
    readability_norm = min(3.0, (flesch / 100) * 3)
    diversity_score  = min(2.0, lexical_diversity * 4)
    length_score     = min(2.0, (word_count / 300) * 2)  
    similarity_score = min(3.0, similarity * 3)
    composite        = round(
        readability_norm + diversity_score + length_score + similarity_score, 2
    )

    return {
        "readability_score":  round(float(flesch), 1),
        "readability_grade":  round(float(fk_grade), 1),
        "word_count":         word_count,
        "sentence_count":     sentence_count,
        "lexical_diversity":  lexical_diversity,
        "similarity_score":   similarity,
        "composite_score":    composite,
    }


def _empty_metrics() -> dict:
    """Return zeroed metrics dict for failed LLM calls."""
    return {
        "readability_score": 0.0,
        "readability_grade": 0.0,
        "word_count":        0,
        "sentence_count":    0,
        "lexical_diversity": 0.0,
        "similarity_score":  0.0,
        "composite_score":   0.0,
    }


def _find_winner(results: list) -> str:
    """Return the provider name with the highest composite score."""
    valid = [r for r in results if not r.get("error", True)]
    if not valid:
        return "N/A"
    best = max(valid, key=lambda r: r["metrics"].get("composite_score", 0))
    return best["provider"]


def generate_project_insights(
    result:          dict,
    raw_text:        str = "",
    chunk_count:     int = 0,
    pdf_name:        str = "",
) -> dict:
    """
    Generates an overall project evaluation dashboard from the
    pipeline result state. All computed locally — zero API calls.

    Evaluates:
        Paper Complexity    — FK grade of raw text
        Summary Quality     — Critic scores from pipeline
        Section Coverage    — Sections detected vs guaranteed 5
        Visual Richness     — Images extracted
        Citation Density    — Citations per estimated page count
        Pipeline Efficiency — Iterations needed to pass critic

    Returns:
        Dict of insight categories, each with score + label + detail.
    """
    insights = {}

    # Note: For paper complexity, we use the raw text (up to 5000 chars) to get a general sense of the paper's writing style and difficulty. This is a rough estimate and may not perfectly reflect the actual reading experience, especially if the paper has a complex structure or uses a lot of domain-specific terminology. However, it provides a useful baseline metric for users to understand the potential difficulty of the paper at a glance.
    if raw_text:
        readability = _compute_readability(raw_text[:5000])
        fk_grade    = readability.get("eval_readability_grade") or 10.0
        flesch      = readability.get("eval_readability_score") or 50.0

        if fk_grade >= 16:
            complexity_label = "Very High (Graduate Level)"
            complexity_score = 5
        elif fk_grade >= 13:
            complexity_label = "High (Undergraduate Level)"
            complexity_score = 4
        elif fk_grade >= 10:
            complexity_label = "Medium (High School Level)"
            complexity_score = 3
        else:
            complexity_label = "Accessible (General Audience)"
            complexity_score = 2

        insights["paper_complexity"] = {
            "score":  complexity_score,
            "max":    5,
            "label":  complexity_label,
            "detail": f"Flesch-Kincaid Grade: {fk_grade} | Flesch Ease: {flesch}",
        }

    # Note: For Summary quality, we rely on the critic scores that were generated during the main pipeline execution. These scores are based on the LLM's evaluation of the summary's accuracy, completeness, and clarity compared to the original paper content. While these are subjective evaluations from the LLM, they provide a useful proxy for the overall quality of the summary. By combining these scores into an overall rating, we can give users a quick sense of how well the summary captures the key information from the paper.
    eval_accuracy     = result.get("eval_accuracy")
    eval_completeness = result.get("eval_completeness")
    eval_clarity      = result.get("eval_clarity")
    eval_overall      = result.get("eval_overall")

    if eval_overall:
        quality_label = (
            "Excellent" if eval_overall >= 4.5 else
            "Good"      if eval_overall >= 3.5 else
            "Adequate"  if eval_overall >= 2.5 else
            "Needs Improvement"
        )
        insights["summary_quality"] = {
            "score":  eval_overall,
            "max":    5.0,
            "label":  quality_label,
            "detail": (
                f"Accuracy: {eval_accuracy}/5 | "
                f"Completeness: {eval_completeness}/5 | "
                f"Clarity: {eval_clarity}/5"
            ),
        }

    # Note: For section coverage, we look at the number of sections that were successfully summarised by the section summarizer agent. This includes both sections that were explicitly detected based on headings in the paper and critical sections that were extracted using the LLM. A higher number of summarised sections indicates better coverage of the paper's content, which can lead to a more comprehensive summary. By categorising the coverage into labels like "Comprehensive", "Good", "Partial", and "Limited", we provide users with an intuitive understanding of how well the pipeline was able to capture the structure and key components of the paper.
    section_summaries = result.get("section_summaries") or {}
    n_sections        = len(section_summaries)

    if n_sections >= 7:
        coverage_label = "Comprehensive (Full paper coverage)"
        coverage_score = 5
    elif n_sections >= 5:
        coverage_label = "Good (All critical sections covered)"
        coverage_score = 4
    elif n_sections >= 3:
        coverage_label = "Partial (Some sections missing)"
        coverage_score = 3
    else:
        coverage_label = "Limited (Minimal coverage)"
        coverage_score = 2

    insights["section_coverage"] = {
        "score":  coverage_score,
        "max":    5,
        "label":  coverage_label,
        "detail": f"{n_sections} sections summarised",
    }

    # Note: For citation density, we count the number of citations extracted by the pipeline and normalise it by an estimated page count of the paper. This gives us a sense of how well the pipeline was able to capture the references and citations within the paper, which are often critical for understanding the context and significance of the research. A higher citation density suggests that the pipeline was able to identify and include more of the relevant literature cited in the paper, which can enhance the depth and credibility of the summary. By categorising citation density into labels like "Rich Bibliography", "Good References", "Moderate References", and "Limited References", we provide users with an intuitive understanding of how well the pipeline captured the scholarly context of the paper.
    citations = result.get("citations") or []
    n_cites   = len(citations)

    est_pages = max(1, len(raw_text) // 3000)
    cite_density = round(n_cites / est_pages, 1)

    if n_cites >= 20:
        cite_label = "Rich Bibliography"
        cite_score = 5
    elif n_cites >= 10:
        cite_label = "Good References"
        cite_score = 4
    elif n_cites >= 5:
        cite_label = "Moderate References"
        cite_score = 3
    else:
        cite_label = "Limited References"
        cite_score = 2

    insights["citations"] = {
        "score":  cite_score,
        "max":    5,
        "label":  cite_label,
        "detail": f"{n_cites} citations extracted (~{cite_density} per page)",
    }

    # Note: For pipeline efficiency, we look at how many iterations the pipeline needed to go through before the summary passed the critic's evaluation. If the summary passed on the first attempt, that's a strong signal of efficiency and understanding, so it gets the highest score. If it took multiple iterations but eventually passed, it indicates that the pipeline was able to learn and improve, which is still good but not perfect. If it never passed and hit the maximum iterations, that suggests significant challenges in summarising the paper effectively. By categorising efficiency into labels like "Perfect", "Good", "Adequate", and "Max iterations reached", we provide users with an intuitive understanding of how smoothly the pipeline was able to generate a satisfactory summary.
    iterations  = result.get("iteration_count", 1)
    is_clear    = result.get("is_hallucination_free", False)

    if iterations == 1 and is_clear:
        eff_label = "Perfect (Passed on first attempt)"
        eff_score = 5
    elif iterations == 2 and is_clear:
        eff_label = "Good (Passed on second attempt)"
        eff_score = 4
    elif is_clear:
        eff_label = "Adequate (Required multiple revisions)"
        eff_score = 3
    else:
        eff_label = "Max iterations reached"
        eff_score = 2

    insights["pipeline_efficiency"] = {
        "score":  eff_score,
        "max":    5,
        "label":  eff_label,
        "detail": (
            f"{iterations} iteration(s) | "
            f"{'Hallucination-free ✓' if is_clear else 'Forced exit'}"
        ),
    }

    # Note: For document size, we simply look at the character count of the raw text and categorise it into "Very Short", "Short", "Medium", and "Large". This gives users a quick sense of the length of the paper, which can be an important factor in how much information there is to summarise and how complex the paper might be. Longer papers often contain more detailed methodologies, results, and discussions, which can make summarisation more challenging but also more rewarding in terms of insights gained. By providing an estimated page count based on character count, we also give users a familiar reference point for understanding the length of the document.
    char_count = len(raw_text)
    if char_count >= 100000:
        size_label = "Large Document (100k+ characters)"
        size_score = 5
    elif char_count >= 30000:
        size_label = "Medium Document"
        size_score = 4
    elif char_count >= 10000:
        size_label = "Short Document"
        size_score = 3
    else:
        size_label = "Very Short Document"
        size_score = 2

    insights["document_size"] = {
        "score":  size_score,
        "max":    5,
        "label":  size_label,
        "detail": (
            f"{char_count:,} characters | "
            f"{chunk_count} chunks | "
            f"~{max(1, char_count // 3000)} pages estimated"
        ),
    }

    # Overall score: weighted average of all categories, normalised to 10
    scores  = [v["score"] for v in insights.values()]
    maxes   = [v["max"]   for v in insights.values()]
    overall = round(sum(s/m for s, m in zip(scores, maxes)) / len(scores) * 10, 1)

    insights["_overall"] = overall   

    return insights