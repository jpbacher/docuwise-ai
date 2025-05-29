import streamlit as st
import os
from dotenv import load_dotenv

from loaders.file_loader import load_uploaded_files
from core.vector_store import build_vector_store
from core.summarizer_chain import get_summarizer_chain
from core.qa_chain import get_qa_chain

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit page config
st.set_page_config(page_title="RAG Assistant", layout="centered")
st.title("📄🧠 Document Assistant")

# Tone selection dropdown
TONE_OPTIONS = [
    "Professional and Friendly",
    "Concise and Factual",
    "Casual and Conversational",
    "Technical and Detailed",
]
selected_tone = st.selectbox("Choose tone", TONE_OPTIONS, index=1)

# Action selection radio: Summarize or Q&A
action = st.radio(
    "Select action", ["Summarize Documents", "Ask a Question"]
)

# File upload
uploaded_files = st.file_uploader(
    "Upload your documents", type=["pdf", "csv", "txt"], accept_multiple_files=True
)

if uploaded_files and action:
    docs = load_uploaded_files(uploaded_files)
    full_text = "\n".join([doc.page_content for doc in docs])

    if action == "Summarize Documents":
        # Summarization flow
        summarizer = get_summarizer_chain()
        with st.spinner("Generating summary..."):
            response = summarizer.invoke({
                "context": full_text,
                "tone": selected_tone,
            })
            summary = response.get("text")
        st.subheader("📋 Summary")
        st.markdown(summary)

    else:
        # Q&A flow
        question = st.text_input("Enter your question:")
        if question:
            vectorstore = build_vector_store(docs)
            qa_chain = get_qa_chain(vectorstore)
            with st.spinner("Generating answer..."):
                response = qa_chain.invoke({
                    "query": question,
                    "tone": selected_tone,
                })
            answer = response.get("result")
            sources = response.get("source_documents", [])

            st.subheader("🔍 Answer")
            st.write(answer)
            st.subheader("📚 Sources")
            for src in sources:
                st.markdown(f"- {src.metadata.get('source')}")
        else:
            st.info("Please enter a question to get an answer.")
