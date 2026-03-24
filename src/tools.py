import os
import re
import logging
import pymupdf4llm
import arxiv

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# Environment & Logging Setup
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Constants for chunking, retrieval, and ArXiv search parameters
CHUNK_SIZE    = 1200   # characters per chunk
CHUNK_OVERLAP = 200    # overlap between chunks to preserve context
TOP_K_RESULTS = 6      # number of chunks to retrieve per query
ARXIV_MAX_RESULTS = 3  # number of related papers to fetch from ArXiv
FAISS_INDEX_DIR    = "data/faiss_index"  # local cache directory for FAISS index files

LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
 
# Embeddings Setup with Fallback 
def _get_embeddings():
    """Local sentence-transformers embeddings — no API quota consumed."""
    try:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
 
        logger.info(f"Using local embeddings: {LOCAL_EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError:
        logger.warning("sentence-transformers not found. Falling back to Google embeddings.")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY not set in .env")
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
 
 
# PDF Ingestion & Processing 
def ingest_pdf(pdf_path: str) -> list[Document]:
    """
    Extract text from PDF using pymupdf4llm and split into chunks.
 
    Args:
        pdf_path: Path to the PDF file.
 
    Returns:
        List of LangChain Document objects.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: '{pdf_path}'")
 
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"Expected .pdf file, got: '{pdf_path}'")
 
    logger.info(f"Extracting text from: {pdf_path}")
    
    try:
        logging.getLogger("pymupdf4llm").setLevel(logging.ERROR)  # suppress verbose logs
        markdown_text: str = pymupdf4llm.to_markdown(pdf_path)
    except Exception as e:
        raise RuntimeError(f"pymupdf4llm failed: {e}") from e
 
    if not markdown_text.strip():
        raise RuntimeError("Extraction returned empty text. PDF may be image-only.")
 
    logger.info(f"Extracted {len(markdown_text):,} characters.")
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
 
    chunks = splitter.create_documents(
        texts=[markdown_text],
        metadatas=[{"source": pdf_path}],
    )
 
    if not chunks:
        raise RuntimeError("Text splitter produced zero chunks.")
 
    logger.info(f"Split into {len(chunks)} chunks.")
    return chunks
 
 
# FAISS Vector Store Creation
def get_retriever(chunks: list[Document]):
    """
    Build a local FAISS vector store from document chunks.
 
    Args:
        chunks: List of Document objects from ingest_pdf().
 
    Returns:
        LangChain VectorStoreRetriever.
    """
    if not chunks:
        raise ValueError("Cannot build vector store from empty chunk list.")
 
    embeddings = _get_embeddings()
    logger.info(f"Building FAISS index from {len(chunks)} chunks...")
 
    try:
        vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    except Exception as e:
        raise RuntimeError(f"FAISS build failed: {e}") from e
 
    logger.info("FAISS index built successfully.")
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS},
    )
 
 
# RAG Retrieval Function
def research_paper(query: str, retriever) -> str:
    """
    Retrieve top-k relevant chunks for a query.
 
    Args:
        query    : Search string.
        retriever: FAISS retriever from get_retriever().
 
    Returns:
        Top-k chunks joined with '---' separator.
    """
    if not query or not query.strip():
        raise ValueError("Query must be non-empty.")
 
    if retriever is None:
        logger.warning("research_paper: retriever is None.")
        return ""
 
    logger.info(f"Retrieving top-{TOP_K_RESULTS} chunks for: '{query[:80]}'")
 
    try:
        docs: list[Document] = retriever.invoke(query)
    except Exception as e:
        logger.warning(f"Retrieval failed: {e}")
        return ""
 
    if not docs:
        return ""
 
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    logger.info(f"Retrieved {len(docs)} chunks ({len(context):,} chars).")
    return context
 
 
# ARXIV Extraction Setup 
def arxiv_search(query: str, max_results: int = ARXIV_MAX_RESULTS) -> list[str]:
    """
    Search ArXiv for related papers. Used for Expert-level enrichment.
 
    Args:
        query      : Search terms.
        max_results: Max papers to return.
 
    Returns:
        List of formatted "Title / Authors / Abstract" strings.
    """
    if not query or not query.strip():
        return []
 
    logger.info(f"Searching ArXiv: '{query[:80]}'")
 
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in search.results():
            entry = (
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(str(a) for a in paper.authors[:3])}\n"
                f"Abstract: {paper.summary[:400]}..."
            )
            results.append(entry)
        logger.info(f"ArXiv returned {len(results)} result(s).")
        return results
    except Exception as e:
        logger.warning(f"ArXiv search failed: {e}")
        return []
 
 
# 5. VISUAL EXTRACTION 

def extract_tables_and_figures(raw_text: str) -> list[str]:
    """
    Layer 1 — Enhanced table and caption extraction.

    Three types of content extracted:
        1. Markdown pipe tables (converted by pymupdf4llm)
        2. Figure/table captions with surrounding context
        3. Data mentioned in caption (numbers, percentages, metrics)

    This gives users meaningful textual understanding of visuals
    even when vision LLM quota is exhausted.
    """
    if not raw_text:
        return []

    results = []
    lines   = raw_text.split("\n")

    # 1. Markdown tables (pymupdf4llm converts simple tables to pipe format)
    table_lines    = []
    pre_context    = []
    in_table       = False

    for i, line in enumerate(lines):
        is_row = "|" in line and line.strip().startswith("|")
        if is_row:
            if not in_table:
                pre_context = [l for l in lines[max(0,i-2):i] if l.strip()]
            in_table = True
            table_lines.append(line)
        else:
            if in_table and len(table_lines) >= 3:
                context_str = " ".join(pre_context).strip()
                table_block = "\n".join(table_lines)
                entry = f"**Table:**"
                if context_str:
                    entry += f" _{context_str[:100]}_\n"
                entry += f"\n{table_block}"
                results.append(entry)
            table_lines = []
            in_table    = False

    # 2. Captions with context (regex search for Fig/Table patterns)
    caption_pattern = re.compile(
        r"((?:Fig(?:ure)?|Table|Fig\.)\s*\.?\s*\d+[:\.\s].{10,300})",
        re.IGNORECASE
    )

    seen = set()
    for match in caption_pattern.finditer(raw_text):
        caption = match.group(1).strip()
        key     = caption[:40]
        if key in seen:
            continue
        seen.add(key)

        # Find surrounding context (text that discusses this figure)
        start_pos  = match.start()
        before     = raw_text[max(0, start_pos - 400):start_pos].strip()
        after      = raw_text[match.end():match.end() + 400].strip()

        # Extract sentences that reference this figure number
        fig_num_match = re.search(r"\d+", caption)
        fig_num       = fig_num_match.group() if fig_num_match else ""

        discussion = ""
        if fig_num:
            discuss_pattern = re.compile(
                rf"(?:Fig(?:ure)?\.?\s*{fig_num}|Table\s*{fig_num})"
                rf"[^.]*\.[^.]*\.",
                re.IGNORECASE
            )
            discuss_matches = discuss_pattern.findall(raw_text)
            if discuss_matches:
                discussion = " ".join(discuss_matches[:2])

        entry = f"**Caption:** {caption}"
        if discussion:
            entry += f"\n**Paper's discussion:** {discussion[:300]}"

        results.append(entry)

    logger.info(f"Extracted {len(results)} table/figure items.")
    return results[:15]


def extract_images_from_pdf(pdf_path: str) -> list[dict]:
    """
    Layer 2 + 3 —> Extract actual images from PDF with surrounding context.

    For each image found:
        - Extracts the raw image bytes
        - Encodes to base64 for LLM vision input
        - Finds the nearest figure caption in surrounding text
        - Returns structured dict ready for vision LLM processing

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts:
        {
            "page":        int,           # page number (1-indexed)
            "index":       int,           # image index on that page
            "caption":     str,           # nearest caption or empty string
            "context":     str,           # surrounding paragraph text
            "base64_data": str,           # base64-encoded PNG bytes
            "media_type":  str,           # always "image/png"
            "width":       int,
            "height":      int,
        }
        Returns empty list if no images found or extraction fails.
    """
    if not pdf_path or not os.path.exists(pdf_path):
        logger.warning(f"extract_images_from_pdf: file not found: {pdf_path}")
        return []

    try:
        import fitz          
        import base64
        from PIL import Image
        import io
    except ImportError:
        logger.warning(
            "pymupdf or pillow not installed. "
            "Run: pip install pymupdf pillow"
        )
        return []

    images    = []
    MIN_SIZE  = 150
    MIN_AREA  = 30000    
    MAX_IMAGES = 12   

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF for image extraction: {e}")
        return []

    for page_num in range(len(doc)):
        page      = doc[page_num]
        page_text = page.get_text("text")   
        img_list  = page.get_images(full=True)

        for img_idx, img_info in enumerate(img_list):
            if len(images) >= MAX_IMAGES:
                break

            xref = img_info[0]

            try:
                base_image  = doc.extract_image(xref)
                img_bytes   = base_image["image"]
                img_width   = base_image["width"]
                img_height  = base_image["height"]

                # Skip tiny images —> likely decorations or icons
                if img_width < MIN_SIZE or img_height < MIN_SIZE:
                    continue
                if img_width * img_height < MIN_AREA:
                    continue    
                # Convert to PNG via Pillow for consistent format
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                buffer  = io.BytesIO()
                pil_img.save(buffer, format="PNG")
                png_bytes   = buffer.getvalue()
                b64_encoded = base64.b64encode(png_bytes).decode("utf-8")

                # Find nearest caption on this page
                caption = _find_nearest_caption(page_text, img_idx)

                # Extract surrounding context (100 chars around caption)
                context = _extract_caption_context(page_text, caption)

                images.append({
                    "page":        page_num + 1,
                    "index":       img_idx,
                    "caption":     caption,
                    "context":     context,
                    "base64_data": b64_encoded,
                    "media_type":  "image/png",
                    "width":       img_width,
                    "height":      img_height,
                })

                logger.info(
                    f"Extracted image: page={page_num+1}, "
                    f"size={img_width}x{img_height}, "
                    f"caption='{caption[:50]}'"
                )

            except Exception as e:
                logger.warning(f"Failed to extract image xref={xref}: {e}")
                continue

    doc.close()
    logger.info(f"Total images extracted: {len(images)}")
    return images


def _find_nearest_caption(page_text: str, img_idx: int) -> str:
    """
    Find the most likely caption for an image on a page.
    Searches for Fig/Figure/Table patterns in the page text.
    Returns the best matching caption or empty string.
    """
    pattern = re.compile(
        r"(?:Fig(?:ure)?|Table|Fig\.)\s*\.?\s*(\d+)[:\.\s]([^\n]{10,200})",
        re.IGNORECASE
    )
    matches = pattern.findall(page_text)

    if not matches:
        return ""

    # Return caption matching image index if possible, else first found
    if img_idx < len(matches):
        num, text = matches[img_idx]
        return f"Figure {num}: {text.strip()}"

    num, text = matches[0]
    return f"Figure {num}: {text.strip()}"


def _extract_caption_context(page_text: str, caption: str) -> str:
    """
    Extract surrounding paragraph text around a caption.
    Gives the vision LLM context about what the figure represents.
    """
    if not caption or not page_text:
        return ""

    # Find caption position and extract surrounding text
    caption_core = caption[:30]
    idx = page_text.find(caption_core)

    if idx == -1:
        return ""

    start = max(0, idx - 200)
    end   = min(len(page_text), idx + 400)
    return page_text[start:end].strip()


# 6. FAISS CACHE

def save_retriever_locally(vector_store, save_path: str = FAISS_INDEX_DIR) -> None:
    """Save FAISS index to disk to avoid re-embedding on next run."""
    os.makedirs(save_path, exist_ok=True)
    try:
        vector_store.save_local(save_path)
        logger.info(f"FAISS index saved to: {save_path}")
    except Exception as e:
        logger.warning(f"Failed to save FAISS index: {e}")
 
 
def load_retriever_locally(save_path: str = FAISS_INDEX_DIR):
    """Load a previously saved FAISS index. Returns None if not found."""
    if not os.path.exists(save_path):
        return None
    try:
        embeddings   = _get_embeddings()
        vector_store = FAISS.load_local(
            save_path, embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"FAISS loaded from cache: {save_path}")
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RESULTS},
        )
    except Exception as e:
        logger.warning(f"Failed to load FAISS cache: {e}")
        return None
 

 
# 7. CITATION EXTRACTION
def extract_citations(raw_text: str) -> list[dict]:
    """
    Extracts references/citations from the raw paper text using regex.
    Handles common reference formats:
        [1] Author et al., "Title," Journal, year.
        [1] Author, A. (2023). Title. Journal.
        IEEE / APA / numbered list styles.

    Args:
        raw_text: Full Markdown text from ingest_pdf().

    Returns:
        List of dicts: [{"number": "1", "text": "full citation string"}, ...]
        Returns empty list if no references section found.
    """
    if not raw_text:
        return []

    citations = []

    # 1. Locate the references section (look for "References", "Bibliography", etc.)
    ref_pattern = re.compile(
        r"(?i)\n(?:#{1,3}\s*)?(?:references|bibliography|works\s+cited|reference\s+list)\s*\n",
        re.IGNORECASE
    )
    match = ref_pattern.search(raw_text)

    if match:
        ref_section = raw_text[match.start():]
    else:
        # Fallback: if no clear section header, take the last 20% of the text as likely references
        cutoff = int(len(raw_text) * 0.80)
        ref_section = raw_text[cutoff:]

    # 2. Extract individual citations using regex patterns
    numbered_bracket = re.findall(
        r"\[(\d{1,3})\]\s+(.{20,400}?)(?=\[\d{1,3}\]|\Z)",
        ref_section,
        re.DOTALL
    )
    if numbered_bracket:
        for num, text in numbered_bracket:
            clean = re.sub(r"\s+", " ", text).strip()
            if len(clean) > 20:
                citations.append({"number": num, "text": clean})

    # Pattern 2 - IEEE / APA style (number at start of line)
    if not citations:
        numbered_dot = re.findall(
            r"(?m)^(\d{1,3})\.\s+(.{20,400}?)(?=^\d{1,3}\.\s|\Z)",
            ref_section,
            re.DOTALL | re.MULTILINE
        )
        for num, text in numbered_dot:
            clean = re.sub(r"\s+", " ", text).strip()
            if len(clean) > 20:
                citations.append({"number": num, "text": clean})

    citations = citations[:50]
    logger.info(f"Extracted {len(citations)} citations.")
    return citations

# PDF

def export_to_pdf(
    summary:                str,
    section_summaries:      dict,
    citations:              list,
    expertise_level:        str,
    pdf_name:               str,
    eval_overall:           float = None,
    eval_accuracy:          int   = None,
    eval_completeness:      int   = None,
    eval_clarity:           int   = None,
    eval_justifications:    dict  = None,
    eval_readability_score: float = None,
    eval_readability_grade: float = None,
    output_path:            str   = "data/scholargraph_report.pdf",
) -> str:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            HRFlowable, Table, TableStyle,
        )
    except ImportError:
        raise ImportError("Run: pip install reportlab")

    # Sanitize inputs and set defaults to avoid None values that break ReportLab
    summary             = summary             or ""
    section_summaries   = section_summaries   or {}
    citations           = citations           or []
    expertise_level     = expertise_level     or "Intermediate"
    pdf_name            = pdf_name            or "paper.pdf"
    eval_justifications = eval_justifications or {}

    output_path = "data/scholargraph_report.pdf"
    import os
    os.makedirs("data", exist_ok=True)

    # Do not attempt to render None values in the PDF —> ReportLab will error out.
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.5 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "SGTitle", parent=styles["Title"],
        fontSize=20, textColor=colors.HexColor("#4a4edb"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "SGSubtitle", parent=styles["Normal"],
        fontSize=11, textColor=colors.HexColor("#555555"),
        spaceAfter=4,
    )
    section_heading_style = ParagraphStyle(
        "SGSectionHeading", parent=styles["Heading2"],
        fontSize=13, textColor=colors.HexColor("#4a4edb"),
        spaceBefore=14, spaceAfter=4,
    )
    summary_heading_style = ParagraphStyle(
        "SGSummaryHeading", parent=styles["Normal"],
        fontSize=11, textColor=colors.HexColor("#222222"),
        fontName="Helvetica-Bold",
        spaceBefore=8, spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "SGBody", parent=styles["Normal"],
        fontSize=10, leading=15,
        textColor=colors.HexColor("#222222"),
        spaceAfter=6,
    )
    table_cell_style = ParagraphStyle(
        "SGTableCell", parent=styles["Normal"],
        fontSize=8, leading=12,
        textColor=colors.HexColor("#333333"),
        wordWrap="CJK",   
    )
    table_header_style = ParagraphStyle(
        "SGTableHeader", parent=styles["Normal"],
        fontSize=9, leading=12,
        textColor=colors.white,
        fontName="Helvetica-Bold",
    )
    citation_style = ParagraphStyle(
        "SGCitation", parent=styles["Normal"],
        fontSize=9, leading=13,
        textColor=colors.HexColor("#444444"),
        leftIndent=16, spaceAfter=4,
    )

    def safe_str(val, fallback="N/A"):
        """Cast to str and strip Unicode that ReportLab cannot render."""
        text = str(val) if val is not None else fallback
        return text.encode("ascii", "ignore").decode("ascii").strip() or fallback

    def stars(s):
        """ASCII star rating — no emoji."""
        if not s:
            return "N/A"
        filled = max(0, min(5, int(round(float(s)))))
        return ("*" * filled) + ("-" * (5 - filled)) + f" ({s}/5)"

    def flesch_label(score):
        if score is None:
            return "N/A"
        score = float(score)
        if score >= 70: return f"{round(score, 1)} (Easy)"
        if score >= 50: return f"{round(score, 1)} (Standard)"
        if score >= 30: return f"{round(score, 1)} (Difficult)"
        return f"{round(score, 1)} (Very Difficult)"

    def is_heading_line(line):
        """
        Detects lines that are section headings in the summary text.
        Matches patterns like: '## Introduction', '**Results**',
        lines ending with ':' that are short, or ALL CAPS short lines.
        """
        s = line.strip()
        if not s or len(s) > 80:
            return False
        # Markdown heading
        if s.startswith("#"):
            return True
        
        if s.endswith(":") and len(s) < 60:
            return True
        
        if s.startswith("**") and s.endswith("**"):
            return True

        if len(s) > 2 and ord(s[0]) > 127:
            return True
        return False

    def clean_heading(line):
        """Strip markdown symbols, return plain heading text."""
        s = line.strip()
        s = s.lstrip("#").strip()
        s = s.strip("**").strip()
        return s.encode("ascii", "ignore").decode("ascii").strip()

    
    story = []

    story.append(Paragraph("ScholarGraph Report", title_style))
    story.append(Paragraph(f"Paper: {safe_str(pdf_name)}", subtitle_style))
    story.append(Paragraph(f"Expertise Level: {safe_str(expertise_level)}", subtitle_style))
    if eval_overall:
        story.append(Paragraph(
            f"Overall Quality: {stars(eval_overall)}",
            subtitle_style
        ))
    story.append(HRFlowable(
        width="100%", thickness=1,
        color=colors.HexColor("#4a4edb"),
        spaceAfter=12
    ))

    if eval_accuracy is not None:
        story.append(Paragraph("Summary Quality Evaluation", section_heading_style))

        def cell(text, header=False):
            style = table_header_style if header else table_cell_style
            return Paragraph(safe_str(text), style)

        eval_rows = [
            
            [cell("Dimension", header=True),
             cell("Score", header=True),
             cell("Justification", header=True)],
            
            [cell("Accuracy"),
             cell(f"{eval_accuracy}/5"),
             cell(eval_justifications.get("Accuracy", "N/A"))],
            [cell("Completeness"),
             cell(f"{eval_completeness}/5"),
             cell(eval_justifications.get("Completeness", "N/A"))],
            [cell("Clarity"),
             cell(f"{eval_clarity}/5"),
             cell(eval_justifications.get("Clarity", "N/A"))],
            [cell("Readability"),
             cell(flesch_label(eval_readability_score)),
             cell(f"Flesch-Kincaid Grade: {eval_readability_grade}")],
            [cell("Overall"),
             cell(f"{eval_overall}/5"),
             cell("Average of Accuracy + Completeness + Clarity")],
        ]

        tbl = Table(
            eval_rows,
            colWidths=[3.2 * cm, 2.8 * cm, 10.5 * cm],  
            repeatRows=1,  
        )
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#4a4edb")),
            ("FONTSIZE",      (0, 0), (-1, -1),  9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),
             [colors.HexColor("#f0f0ff"), colors.white]),
            ("GRID",          (0, 0), (-1, -1),  0.5, colors.HexColor("#cccccc")),
            ("FONTNAME",      (0, -1), (-1, -1), "Helvetica-Bold"),
            ("TOPPADDING",    (0, 0), (-1, -1),  6),
            ("BOTTOMPADDING", (0, 0), (-1, -1),  6),
            ("VALIGN",        (0, 0), (-1, -1),  "TOP"),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.4 * cm))
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.HexColor("#cccccc"),
            spaceAfter=6
        ))

    
    story.append(Paragraph("Adaptive Summary", section_heading_style))

    for line in summary.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if is_heading_line(stripped):
            heading_text = clean_heading(stripped)
            if heading_text:
                story.append(Paragraph(heading_text, summary_heading_style))
        else:
            clean = stripped.encode("ascii", "ignore").decode("ascii").strip()
            if clean:
                story.append(Paragraph(clean, body_style))

    story.append(Spacer(1, 0.3 * cm))

    if section_summaries:
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.HexColor("#cccccc"),
            spaceAfter=6
        ))
        story.append(Paragraph("Section-by-Section Breakdown", section_heading_style))

        for sec_name, sec_text in section_summaries.items():
            safe_name = safe_str(sec_name)
            story.append(Paragraph(safe_name, section_heading_style))
            for line in str(sec_text).split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                clean = stripped.encode("ascii", "ignore").decode("ascii").strip()
                if clean:
                    story.append(Paragraph(clean, body_style))

    if citations:
        story.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.HexColor("#cccccc"),
            spaceAfter=6
        ))
        story.append(Paragraph("References", section_heading_style))
        for cite in citations:
            num  = safe_str(cite.get("number", "?"))
            text = safe_str(cite.get("text", ""))
            story.append(Paragraph(f"[{num}] {text}", citation_style))

    doc.build(story)
    logger.info(f"PDF report saved to: {output_path}")
    return output_path