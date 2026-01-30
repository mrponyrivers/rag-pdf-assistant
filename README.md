cd /Users/ponyrivers/ai-journey/rag-pdf-assistant
cat > README.md <<'MD'
# RAG PDF Assistant (Local)

Local-only PDF Q&A assistant built with Streamlit.  
Upload a PDF → chunk text → embed with Sentence Transformers → retrieve top sources with similarity scores → generate an offline “local synthesis” answer + page citations.

## Project highlights
- **Cached indexing** by PDF hash + chunk settings (fast reloads)
- **Retrieval transparency** with similarity scores + page citations
- **Exports**: `answers.md`, `answers.csv`, and `sources.csv`

## Features
- Upload a PDF and build a local index (no FAISS)
- Top-K retrieval with similarity scores
- Optional: unique pages only + max chunks per page
- Clean Answer vs Sources layout + sources table
- Export: `answers.md`, `answers.csv`, `sources.csv`

## Setup
```bash
cd /Users/ponyrivers/ai-journey/rag-pdf-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```
