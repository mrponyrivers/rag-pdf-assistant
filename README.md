# RAG PDF Assistant (Local)
Instant demo: click 📎 Load sample PDF (no upload required).
## Live demo
https://mrponyrivers-rag-pdf-assistant.streamlit.app/

### What to try
- Upload any PDF and ask: **"What is this document about?"**
- Ask: **"List the key points with page citations."**

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

## Screenshots

### PDF loaded
![PDF Loaded](assets/ui_loaded.png)

### Results
![Results](assets/ui_results.png)

## Tech stack
- Python + Streamlit (UI)
- PDF text extraction
- Sentence-Transformers embeddings
- Similarity search + transparent sources (scores + page citations)

## Setup
```bash
cd /Users/ponyrivers/ai-journey/rag-pdf-assistant
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
