# RAG PDF Assistant (Local)

Local-only PDF Q&A assistant built with Streamlit.
Upload a PDF → chunk text → embed with Sentence Transformers → retrieve top sources with similarity scores → generate an offline “local synthesis” answer + page citations.

## Features
- Upload PDF and build a local index (no FAISS)
- Top-K retrieval with similarity scores
- Optional: Unique pages only + Max chunks per page
- Clean Answer vs Sources layout + sources table
- Export: answers.md, answers.csv, sources.csv
- Cached indexing per PDF hash + settings + Rebuild Index button

## Setup
```bash
cd /Users/ponyrivers/ai-journey/rag-pdf-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

## 2) Add 2 short sections (optional but recommended)
Add these at the bottom of README.md:

```md
## Notes
- Works best with PDFs that contain selectable text (not scanned images).
- Answer is deterministic local synthesis (no external LLM).

## Output files
- `answers.md` (Q&A export)
- `answers.csv` (question, answer summary, pages cited)
- `sources.csv` (latest retrieved chunks)




```
