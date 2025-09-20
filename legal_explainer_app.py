# legal_explainer_app.py
import os
import re
import io
import shutil
import platform
from auth import auth_ui  
import base64
from typing import List, Tuple
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Optional local generation (transformers). We import safely and handle missing deps.
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# ---------------------- Simple user auth ----------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

auth_ui()
if st.session_state["logged_in"]:
    st.write(f"Welcome {st.session_state['user_email']}! You are now on the main app page.")
# ---------------------- Tesseract Auto-Setup ----------------------
def setup_tesseract():
    system = platform.system()
    if system == "Windows":
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True
        return False
    else:
        if shutil.which("tesseract"):
            return True
        return False

if not setup_tesseract():
    raise RuntimeError(
        "❌ Tesseract OCR not found.\n"
        "Please install it:\n"
        "  • Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
        "  • Linux: sudo apt-get install tesseract-ocr\n"
        "  • Mac: brew install tesseract\n"
    )

# ---------------------- Configuration ----------------------
HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_MODEL = "facebook/bart-large-cnn"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

FAISS_INDEX = None
CLAUSES: List[str] = []
CLAUSE_EMBEDDINGS = None

PENALTY_PATTERNS = [r"\bpenalt(y|ies)\b", r"\bfine(s)?\b", r"\bcharge(s)?\b", r"\blast payment", r"late fee"]
AUTO_RENEW_PATTERNS = [r"auto-?renew", r"automatic renewal", r"renew(s)? automatically", r"renewal term"]
TERMINATION_PATTERNS = [r"terminate|termination|cancel(ation|led)?|breach", r"notice period", r"cure period"]

CLAUSE_MIN_CHARS = 40
MAX_CHUNK_CHARS = 800

# ---------------------- Utilities ----------------------
def extract_text_from_pdf(file_stream) -> str:
    try:
        file_stream.seek(0)
    except Exception:
        pass
    file_bytes = file_stream.read()
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    texts = []
    for page in doc:
        text = page.get_text("text")
        if text and text.strip():
            texts.append(text)
        else:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
            texts.append(text)
    return "\n".join(texts)

def extract_text_from_image(file_stream) -> str:
    try:
        file_stream.seek(0)
    except Exception:
        pass
    image = Image.open(file_stream).convert("RGB")
    text = pytesseract.image_to_string(image)
    return text

def clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def split_into_clauses(text: str) -> List[str]:
    text = text.replace("\r", "\n")
    chunks = re.split(r"\n\s*\n+", text)
    clauses: List[str] = []
    for c in chunks:
        c = clean_whitespace(c)
        if len(c) < CLAUSE_MIN_CHARS:
            continue
        if len(c) > MAX_CHUNK_CHARS:
            parts = re.split(r"(?m)(?=^\s*\d+\.|^\s*\([a-zA-Z0-9]+\))", c)
            sub: List[str] = []
            for p in parts:
                p = clean_whitespace(p)
                if not p:
                    continue
                if len(p) > MAX_CHUNK_CHARS:
                    sents = re.split(r"(?<=[.!?])\s+", p)
                    bucket = ""
                    for s in sents:
                        if len(bucket) + len(s) < MAX_CHUNK_CHARS:
                            bucket += " " + s
                        else:
                            sub.append(clean_whitespace(bucket))
                            bucket = s
                    if bucket:
                        sub.append(clean_whitespace(bucket))
                else:
                    sub.append(p)
            for s in sub:
                if len(s) >= CLAUSE_MIN_CHARS:
                    clauses.append(s)
        else:
            clauses.append(c)
    final: List[str] = []
    seen = set()
    for cl in clauses:
        cl = cl.strip()
        if cl and cl not in seen:
            final.append(cl)
            seen.add(cl)
    return final

def detect_risks_for_clause(clause: str) -> List[str]:
    flags: List[str] = []
    low = clause.lower()
    for p in PENALTY_PATTERNS:
        if re.search(p, low):
            flags.append("Penalty/Fee")
            break
    for p in AUTO_RENEW_PATTERNS:
        if re.search(p, low):
            flags.append("Auto-Renewal")
            break
    for p in TERMINATION_PATTERNS:
        if re.search(p, low):
            flags.append("Termination/Cancellation")
            break
    return flags

# ---------------------- Summarization ----------------------
_local_summarizer = None
models_to_try = ["sshleifer/distilbart-cnn-12-6", "facebook/bart-large-cnn"]
try:
    _local_summarizer = pipeline("summarization", model="t5-small")
except Exception:
    _local_summarizer = None
for m in models_to_try:
    try:
        _local_summarizer = pipeline("summarization", model=m)
        break
    except Exception:
        pass

def generate_text(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    if HF_API_KEY:
        headers = {"Authorization": f"Bearer {HF_API_KEY}", "Accept": "application/json"}
        api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": temperature}}
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
                return data[0]["generated_text"]
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]
            if isinstance(data, list) and len(data) > 0:
                return str(data[0].get("generated_text", data[0]))
            return str(data)
        except:
            pass
    if _local_summarizer:
        try:
            out = _local_summarizer(prompt, max_length=max_tokens, min_length=30, do_sample=False)
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict) and "summary_text" in out[0]:
                return out[0]["summary_text"]
            return str(out)
        except:
            pass
    sents = re.split(r"(?<=[.!?])\s+", prompt)
    return " ".join(sents[:3] + (sents[-2:] if len(sents) > 3 else []))[:4000]

# ---------------------- Embedding & FAISS ----------------------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

def build_faiss_index(clauses: List[str]):
    global FAISS_INDEX, CLAUSE_EMBEDDINGS
    model = load_embed_model()
    embeddings = model.encode(clauses, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatIP(EMBED_DIM)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    FAISS_INDEX = index
    CLAUSE_EMBEDDINGS = embeddings
    return index

def retrieve_relevant_clauses(question: str, top_k: int = 4) -> List[Tuple[int, float]]:
    model = load_embed_model()
    q_emb = model.encode([question], show_progress_bar=False)
    q_emb = np.array(q_emb).astype("float32")
    faiss.normalize_L2(q_emb)
    if FAISS_INDEX is None:
        return []
    D, I = FAISS_INDEX.search(q_emb, top_k)
    results: List[Tuple[int, float]] = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        results.append((int(idx), float(score)))
    return results

# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Legal Explainer", layout="wide")
st.title(f"📄 Legal Explainer — Contract Summariser & Q&A (User: {st.session_state.user_email})")

# --- Sidebar upload/options ---
st.sidebar.header("Upload Contract")
uploaded = st.sidebar.file_uploader(
    "Upload PDF or image (png/jpg). Multiple pages supported.",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=False
)
if uploaded is None:
    st.sidebar.info("Upload a PDF or scanned image of a contract to begin.")

with st.sidebar.expander("Options"):
    summarise_using_api = st.checkbox("Attempt using Hugging Face Inference API", value=bool(HF_API_KEY))
    top_k = st.number_input("Number of retrieved clauses for Q&A", min_value=1, max_value=10, value=4)
    show_raw_text = st.checkbox("Show extracted raw text", value=False)

raw_text = ""
if uploaded is not None:
    filename = uploaded.name.lower()
    try:
        if filename.endswith(".pdf"):
            raw_text = extract_text_from_pdf(uploaded)
        else:
            raw_text = extract_text_from_image(uploaded)
        raw_text = clean_whitespace(raw_text)
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        raw_text = ""

    if show_raw_text and raw_text:
        st.sidebar.subheader("Raw extracted text (first 2000 chars)")
        st.sidebar.text(raw_text[:2000])

    CLAUSES = split_into_clauses(raw_text) if raw_text else []
    if len(CLAUSES) == 0 and raw_text:
        st.warning("No clauses detected — document may be too short or OCR failed.")
    if CLAUSES:
        with st.spinner("Building semantic index (FAISS) ..."):
            try:
                build_faiss_index(CLAUSES)
            except Exception as e:
                st.error(f"Failed to build semantic index: {e}")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Summary", "Clauses + Risk Flags", "Ask Questions"])

with tab1:
    st.header("Summary")
    if not uploaded:
        st.info("Upload a contract to generate a summary.")
    else:
        summ_prompt = f"Summarize contract:\n{raw_text[:4000]}"
        with st.spinner("Generating summary..."):
            summary = generate_text(summ_prompt, max_tokens=200)

        st.subheader("Plain-language summary")
        st.write(summary)

        st.subheader("Clause-by-clause explanations")
        max_default = min(10, len(CLAUSES)) if CLAUSES else 1
        explain_slider = st.slider(
            "How many clauses to explain",
            min_value=1,
            max_value=max_default,
            value=min(3, max_default)
        )

        clause_explanations = []
        for i in range(explain_slider):
            cl = CLAUSES[i]
            st.markdown(f"**Clause {i+1}:** {cl[:300]}{'...' if len(cl) > 300 else ''}")
            expl_prompt = f"Simplify this clause into bullets:\n{cl}\nPlain language:"
            expl = generate_text(expl_prompt, max_tokens=120)
            st.write(expl)
            clause_explanations.append((cl, expl))

        # PDF Export
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        if st.button("📥 Export summary as PDF"):
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            # Title
            story.append(Paragraph("Contract Summary - Legal Explainer", styles['Heading1']))
            story.append(Spacer(1, 12))

            # Summary text
            story.append(Paragraph(summary.replace("\n", "<br/>"), styles['Normal']))
            story.append(Spacer(1, 12))

            # Clause explanations
            for i, (cl, expl) in enumerate(clause_explanations, start=1):
                story.append(Paragraph(f"<b>Clause {i}:</b> {cl}", styles['Normal']))
                story.append(Paragraph(expl.replace("\n", "<br/>"), styles['Normal']))
                story.append(Spacer(1, 12))

            # Build PDF
            doc.build(story)
            buffer.seek(0)

            # Download button
            st.download_button(
                label="📥 Download Summary PDF",
                data=buffer,
                file_name="contract_summary.pdf",
                mime="application/pdf"
            )

with tab2:
    st.header("Clauses & Risk Flags")
    if not uploaded:
        st.info("Upload a document to view clauses and detected risks.")
    else:
        st.write(f"Detected {len(CLAUSES)} clauses.")
        for i, cl in enumerate(CLAUSES):
            flags = detect_risks_for_clause(cl)
            cols = st.columns([0.9, 0.1])
            with cols[0]:
                st.markdown(f"**Clause {i+1}**: {cl}")
            with cols[1]:
                if flags:
                    st.warning(", ".join(flags))
                else:
                    st.write("")

with tab3:
    st.header("Ask Questions — Contract Q&A")
    if not uploaded:
        st.info("Upload a contract to use the Q&A chatbot.")
    else:
        user_q = st.text_area("Ask a question about this contract")
        if st.button("Get Answer") and user_q.strip():
            with st.spinner("Retrieving relevant clauses..."):
                results = retrieve_relevant_clauses(user_q, top_k=top_k)
            if not results:
                st.info("No semantic index available or no clauses indexed.")
            else:
                cited_text = "\n\n".join([f"Clause {idx+1}: {CLAUSES[idx]}" for idx, _ in results])
                qa_prompt = f"User question: {user_q}\nRelevant clauses:\n{cited_text}\nAnswer concisely."
                with st.spinner("Generating answer..."):
                    answer = generate_text(qa_prompt, max_tokens=300)
                st.subheader("Answer")
                st.write(answer)
                st.subheader("Cited clauses")
                for idx, score in results:
                    st.markdown(f"**Clause {idx+1}** (score: {score:.3f}) — {CLAUSES[idx]}")

# Footer notes
st.markdown("---")
st.write(
    "**Notes & Limitations:** This demo uses heuristics and regex for clause splitting and risk detection. "
    "For production-grade legal review, use licensed counsel. Hugging Face API is attempted if key present."
)

# Sidebar info
if HF_API_KEY:
    st.sidebar.success("Hugging Face API key present.")
else:
    st.sidebar.info("No Hugging Face API key found — using local fallbacks.")
if TRANSFORMERS_AVAILABLE and _local_summarizer:
    st.sidebar.success("Local summarizer available.")
elif TRANSFORMERS_AVAILABLE:
    st.sidebar.warning("Transformers installed but summarizer failed to load.")
else:
    st.sidebar.info("Transformers not installed — local summarization disabled.")
