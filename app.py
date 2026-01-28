import io
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Chunk:
    text: str
    page: int
    chunk_id: int


# -----------------------------
# Helpers
# -----------------------------
def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def normalize_ws(s: str) -> str:
    return " ".join((s or "").split())


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start = max(0, end - overlap)

    return chunks


def extract_pdf_chunks(pdf_bytes: bytes, chunk_size: int, overlap: int) -> Tuple[int, List[Chunk]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    num_pages = len(reader.pages)

    chunks: List[Chunk] = []
    cid = 0

    for page_idx, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        txt = normalize_ws(txt)
        if not txt:
            continue

        for ch in chunk_text(txt, chunk_size=chunk_size, overlap=overlap):
            chunks.append(Chunk(text=ch, page=page_idx, chunk_id=cid))
            cid += 1

    return num_pages, chunks


@st.cache_resource
def load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embs = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(embs, dtype=np.float32)


@st.cache_data(show_spinner=False)
def build_local_index(
    pdf_bytes: bytes,
    model_name: str,
    chunk_size: int,
    overlap: int,
) -> Tuple[int, List[Dict], np.ndarray]:
    """
    Cached by Streamlit automatically based on pdf_bytes + settings.
    Returns:
      - num_pages
      - chunks_serialized: list[dict] for cache friendliness
      - chunk_embs: np.ndarray
    """
    num_pages, chunks = extract_pdf_chunks(pdf_bytes, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return num_pages, [], np.zeros((0, 1), dtype=np.float32)

    embedder = load_embedder(model_name)
    chunk_embs = embed_texts(embedder, [c.text for c in chunks])

    chunks_serialized = [{"text": c.text, "page": c.page, "chunk_id": c.chunk_id} for c in chunks]
    return num_pages, chunks_serialized, chunk_embs


def retrieve_with_scores(query_emb: np.ndarray, chunk_embs: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
    if chunk_embs is None or len(chunk_embs) == 0:
        return []

    sims = chunk_embs @ query_emb  # cosine similarity because normalized
    top_k = min(top_k, int(sims.shape[0]))
    if top_k <= 0:
        return []

    idx = np.argpartition(-sims, top_k - 1)[:top_k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


def apply_source_filters(
    ranked: List[Tuple[int, float]],
    chunks: List[Chunk],
    unique_pages_only: bool,
    max_chunks_per_page: int,
) -> List[Tuple[int, float]]:
    if not ranked:
        return []

    out: List[Tuple[int, float]] = []
    page_counts: Dict[int, int] = {}
    seen_pages: set[int] = set()

    for idx, score in ranked:
        page = chunks[idx].page

        if unique_pages_only and page in seen_pages:
            continue

        if max_chunks_per_page > 0:
            page_counts.setdefault(page, 0)
            if page_counts[page] >= max_chunks_per_page:
                continue
            page_counts[page] += 1

        out.append((idx, score))
        seen_pages.add(page)

    return out


def build_answer_from_sources(question: str, sources: List[Chunk]) -> str:
    # Local-only deterministic “synthesis” (no external LLM)
    lines: List[str] = []
    lines.append(f"### Question\n{question}\n")
    lines.append("### Answer (local synthesis)\n")

    for i, ch in enumerate(sources, start=1):
        parts = ch.text.replace("?", ".").replace("!", ".").split(".")
        short = ". ".join([p.strip() for p in parts if p.strip()][:2]).strip()
        if short and not short.endswith("."):
            short += "."
        lines.append(f"- ({i}) p.{ch.page}: {short}")

    lines.append("\n### Citations\n")
    pages = sorted({c.page for c in sources})
    lines.append("Pages cited: " + ", ".join(map(str, pages)) if pages else "Pages cited: (none)")
    return "\n".join(lines).strip() + "\n"


def answers_to_markdown(history: List[Dict]) -> str:
    out: List[str] = []
    out.append("# RAG PDF Assistant — Q&A Export")
    out.append(f"- Exported: {datetime.now().isoformat(timespec='seconds')}\n")
    for i, item in enumerate(history, start=1):
        out.append(f"## {i}) {item['question']}\n")
        out.append(item["answer_md"].strip())
        out.append("\n---\n")
    return "\n".join(out)


def history_to_csv(history: List[Dict]) -> pd.DataFrame:
    rows = []
    for item in history:
        pages = sorted({c.page for c in item.get("sources", [])})
        rows.append(
            {
                "question": item.get("question", ""),
                "answer_summary": item.get("answer_summary", ""),
                "pages_cited": ",".join(map(str, pages)),
            }
        )
    return pd.DataFrame(rows)


def sources_to_csv(latest_sources: List[Dict]) -> pd.DataFrame:
    if not latest_sources:
        return pd.DataFrame(columns=["rank", "page", "score", "chunk_id", "chunk_preview", "chunk_text"])
    return pd.DataFrame(
        [
            {
                "rank": s["rank"],
                "page": s["page"],
                "score": s["score"],
                "chunk_id": s["chunk_id"],
                "chunk_preview": s["preview"],
                "chunk_text": s["text"],
            }
            for s in latest_sources
        ]
    )


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="RAG PDF Assistant (Local)", page_icon="📄", layout="wide")
st.title("📄 RAG PDF Assistant (Local)")
st.caption("Upload a PDF → ask questions → get local answers with cited pages. (No FAISS, no external LLM.)")

# Session state defaults
if "history" not in st.session_state:
    st.session_state.history = []
if "pdf_id" not in st.session_state:
    st.session_state.pdf_id = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "num_pages" not in st.session_state:
    st.session_state.num_pages = 0
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_embs" not in st.session_state:
    st.session_state.chunk_embs = None
if "latest_sources_table" not in st.session_state:
    st.session_state.latest_sources_table = []
if "force_rebuild" not in st.session_state:
    st.session_state.force_rebuild = False

with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox(
        "Embedding model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
        ],
        index=0,
    )

    chunk_size = st.slider("Chunk size (chars)", 400, 1600, 900, 50)
    overlap = st.slider("Overlap (chars)", 0, 400, 150, 10)

    st.divider()
    st.subheader("Retrieval controls")
    top_k = st.slider("Top-K retrieved chunks", 1, 20, 8, 1)
    unique_pages_only = st.toggle("Unique pages only", value=False)
    max_chunks_per_page = st.slider("Max chunks per page", 1, 10, 3, 1)

    st.divider()
    if st.button("Rebuild index", use_container_width=True):
        st.cache_data.clear()
        st.session_state.force_rebuild = True
        st.session_state.history = []
        st.session_state.latest_sources_table = []
        st.rerun()

    st.caption("Tip: If results feel scattered, turn on **Unique pages only** or lower **Max chunks/page**.")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if not uploaded:
    st.info("Upload a PDF to begin.")
    st.stop()

pdf_bytes = uploaded.getvalue()
pdf_id = file_hash(pdf_bytes)
pdf_name = getattr(uploaded, "name", "uploaded.pdf")

# Determine rebuild
settings_key = (model_name, chunk_size, overlap)
prev_key = st.session_state.get("settings_key")

needs_rebuild = (
    st.session_state.pdf_id != pdf_id
    or prev_key != settings_key
    or st.session_state.chunk_embs is None
    or st.session_state.force_rebuild
)

if needs_rebuild:
    with st.spinner("Processing PDF and building local index..."):
        num_pages, chunks_serialized, chunk_embs = build_local_index(
            pdf_bytes=pdf_bytes,
            model_name=model_name,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        if len(chunks_serialized) == 0:
            st.error("No text extracted from this PDF. Try a different PDF (or one with selectable text).")
            st.stop()

        st.session_state.pdf_id = pdf_id
        st.session_state.pdf_name = pdf_name
        st.session_state.num_pages = num_pages
        st.session_state.chunks = [Chunk(**d) for d in chunks_serialized]
        st.session_state.chunk_embs = chunk_embs
        st.session_state.settings_key = settings_key
        st.session_state.history = []
        st.session_state.latest_sources_table = []
        st.session_state.force_rebuild = False

# PDF loaded panel
left, right = st.columns([1.2, 1])
with left:
    st.success("PDF loaded ✅")
    st.write(
        f"**File:** {st.session_state.pdf_name}\n\n"
        f"**PDF ID:** `{st.session_state.pdf_id}`\n\n"
        f"**Pages:** {st.session_state.num_pages}\n\n"
        f"**Chunks:** {len(st.session_state.chunks)}"
    )

with right:
    st.markdown("### Current settings")
    st.write(
        f"- **Model:** `{model_name}`\n"
        f"- **Chunk size:** {chunk_size}\n"
        f"- **Overlap:** {overlap}\n"
        f"- **Top-K:** {top_k}\n"
        f"- **Unique pages only:** {'Yes' if unique_pages_only else 'No'}\n"
        f"- **Max chunks/page:** {max_chunks_per_page}"
    )

st.divider()

# Input + exports
st.subheader("Ask a question")
question = st.text_input("Type your question", placeholder="e.g., What are the key terms and deadlines?")

c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.2])
with c1:
    ask = st.button("Ask", type="primary", use_container_width=True)
with c2:
    clear = st.button("Clear chat", use_container_width=True)
with c3:
    export_md = st.button("Export answers.md", use_container_width=True)
with c4:
    export_csv = st.button("Export answers.csv", use_container_width=True)

if clear:
    st.session_state.history = []
    st.session_state.latest_sources_table = []
    st.rerun()

if export_md:
    md = answers_to_markdown(st.session_state.history)
    st.download_button(
        "Download answers.md",
        data=md.encode("utf-8"),
        file_name="answers.md",
        mime="text/markdown",
        use_container_width=True,
    )

if export_csv:
    df = history_to_csv(st.session_state.history)
    st.download_button(
        "Download answers.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="answers.csv",
        mime="text/csv",
        use_container_width=True,
    )

with st.expander("Export latest retrieved sources (CSV)"):
    if st.session_state.latest_sources_table:
        df_sources = sources_to_csv(st.session_state.latest_sources_table)
        st.download_button(
            "Download sources.csv",
            data=df_sources.to_csv(index=False).encode("utf-8"),
            file_name="sources.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.caption("Ask a question first to generate retrieved sources.")

# Retrieval
if ask and question.strip():
    with st.spinner("Searching PDF..."):
        embedder = load_embedder(model_name)
        q_emb = embed_texts(embedder, [question])[0]

        ranked = retrieve_with_scores(q_emb, st.session_state.chunk_embs, top_k=top_k)
        ranked = apply_source_filters(
            ranked=ranked,
            chunks=st.session_state.chunks,
            unique_pages_only=unique_pages_only,
            max_chunks_per_page=max_chunks_per_page,
        )

        sources: List[Chunk] = [st.session_state.chunks[i] for i, _ in ranked]

        latest_sources_table = []
        for r, (idx, score) in enumerate(ranked, start=1):
            ch = st.session_state.chunks[idx]
            preview = ch.text[:160].strip() + ("…" if len(ch.text) > 160 else "")
            latest_sources_table.append(
                {
                    "rank": r,
                    "page": ch.page,
                    "score": score,
                    "chunk_id": ch.chunk_id,
                    "preview": preview,
                    "text": ch.text,
                }
            )
        st.session_state.latest_sources_table = latest_sources_table

        answer_md = build_answer_from_sources(question, sources)

        # short summary for CSV (first sentence from up to 3 sources)
        summary_parts: List[str] = []
        for ch in sources[:3]:
            parts = ch.text.replace("?", ".").replace("!", ".").split(".")
            short = ". ".join([p.strip() for p in parts if p.strip()][:1]).strip()
            if short:
                summary_parts.append(short)
        answer_summary = " | ".join(summary_parts)[:280]

        st.session_state.history.insert(
            0,
            {
                "question": question,
                "answer_md": answer_md,
                "answer_summary": answer_summary,
                "sources": sources,
                "sources_table": latest_sources_table,
            },
        )

st.divider()

# Display: clean Answer vs Sources
st.subheader("Results")

if not st.session_state.history:
    st.write("No questions yet. Ask one above 👆")
else:
    item = st.session_state.history[0]
    ans_col, src_col = st.columns([1.15, 1])

    with ans_col:
        st.markdown("## ✅ Answer")
        st.markdown(item["answer_md"])

    with src_col:
        st.markdown("## 🔎 Retrieved sources")
        df = pd.DataFrame(item.get("sources_table", []))
        if not df.empty:
            df_display = df[["rank", "page", "score", "chunk_id", "preview"]].copy()
            df_display["score"] = df_display["score"].map(lambda x: round(float(x), 4))
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.caption("No sources yet.")

        with st.expander("Show full source chunks"):
            for s in item.get("sources_table", []):
                st.markdown(
                    f"**Rank {s['rank']} • Page {s['page']} • Score {s['score']:.4f} • Chunk {s['chunk_id']}**"
                )
                st.write(s["text"])
                st.divider()

    st.markdown("### Previous questions")
    for prev in st.session_state.history[1:]:
        with st.expander(f"Q: {prev['question']}"):
            p_ans, p_src = st.columns([1.15, 1])
            with p_ans:
                st.markdown("#### Answer")
                st.markdown(prev["answer_md"])
            with p_src:
                st.markdown("#### Retrieved sources")
                dfp = pd.DataFrame(prev.get("sources_table", []))
                if not dfp.empty:
                    dfp2 = dfp[["rank", "page", "score", "chunk_id", "preview"]].copy()
                    dfp2["score"] = dfp2["score"].map(lambda x: round(float(x), 4))
                    st.dataframe(dfp2, use_container_width=True, hide_index=True)
                with st.expander("Show full source chunks"):
                    for s in prev.get("sources_table", []):
                        st.markdown(
                            f"**Rank {s['rank']} • Page {s['page']} • Score {s['score']:.4f} • Chunk {s['chunk_id']}**"
                        )
                        st.write(s["text"])
                        st.divider()

 
