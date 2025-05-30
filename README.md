# Document Assistant: 📄🧠

An interactive Streamlit application that leverages LangChain and OpenAI to transform documents into meaningful insights—whether through concise, tone-driven summaries or precise, source-backed answers.

---

## 🚀 Overview
**Document Assistant** is a beginner-friendly Retrieval-Augmented Generation (RAG) project demonstrating key AI engineering patterns:

- **Summarization**: Generate bullet-point summaries tailored by tone and length.  
- **Q&A**: Ask questions about your document and receive accurate, context-grounded answers with citation transparency.

This project highlights best practices in prompt templating, chain composition, vector indexing, and interactive UI design.

---

## 🎯 Real-World Use Cases

- **Knowledge Workers & Analysts**: Distill lengthy reports, research papers, or meeting transcripts into digestible bullet points.  
- **Recruiters & HR**: Summarize resumes and extract key candidate qualifications.  
- **Compliance & Legal Teams**: Query regulatory documents and receive verifiable answers.  
- **Students & Educators**: Summarize lecture notes or textbooks and clarify concepts.

---

## 🏗 Architecture & Components

### 1. Streamlit App (`app.py`)
- **UI Controls**: File uploader, tone selector, action radio (Summarize vs. Q&A).  
- **Orchestration**: Routes inputs to the appropriate LLM chain and displays results.

### 2. Document Loader (`loaders/file_loader.py`)
- **File Parsing**: Supports PDF, CSV, TXT via LangChain loaders.  
- **Metadata Override**: Preserves original filenames for clean source citations.

### 3. Summarization Chain (`core/summarizer_chain.py`)
- **PromptTemplate**: `SUMMARIZATION_PROMPT` defines tone, bullet-count guidance, and anti-hallucination guards.  
- **LLMChain**: Wraps OpenAI’s GPT-3.5-turbo with controlled temperature for consistency.

### 4. Q&A Chain (`core/qa_chain.py`)
- **Vector Store**: Chroma DB–backed embeddings for fast, scalable retrieval. 
- **PromptTemplate**: `QA_PROMPT` enforces answer fidelity and tone.  
- **Chain Execution**: Combines retrieval with generation for accurate responses.

---

## 🛠 Installation

**Clone the repo**  
   ```bash
   git clone https://github.com/jpbacher/docuwise-ai.git
   cd docuwise-ai
```
**Create a Virtual Environment**
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
**Install Dependencies**
```bash
pip install -r requirements.txt
```
**Set your OpenAI API Key**: Create a file named .env in the project root with the following content:
```bash
echo "OPENAI_API_KEY=your_api_key" > .env
```
**Run the Streamlit app**
```bash
streamlit run app.py
```
1. Your browser will open at http://localhost:8501.

2. Upload your document.

3. Choose a tone.

4. Select “Summarize Document” or “Ask a Question.”

5. View the output directly in the UI.

## 🌱 Next Steps & Enhancement

1. **User Feedback Loop**: Collect user ratings on summaries/answers to fine-tune prompts or LLM parameters.

2. **Multi-Document Support**: Batch-process folders with labeled summaries per file.

3. **UI/UX Improvements**: Pagination for long outputs, download buttons, theme customization.

4. **Authentication & Deployment**: Secure with OAuth and deploy via Docker on AWS, Azure, or Streamlit Cloud.
