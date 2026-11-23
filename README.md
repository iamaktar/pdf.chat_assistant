# 📄 PDF Chat Assistant

A Python Retrieval-Augmented Generation (RAG) web app built with Streamlit. Upload multiple PDFs, ask follow-up questions with chat memory, and get streamed answers with exact filename and page citations.

## Tech Stack
- **UI:** Streamlit
- **LLMs & Embeddings:** Google GenAI (`google-genai`) / OpenAI (`openai`)
- **PDF Parsing:** `pypdf`
- **Vector Search:** `numpy` (Cosine Similarity)

## Run Locally
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Add your GOOGLE_API_KEY in .env
streamlit run app.py