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
    "Select action", ["Summarize Document", "Ask a Question"]
)

# Single file upload
uploaded_file = st.file_uploader(
    "Upload a document to process",
    type=["pdf", "csv", "txt"], 
    accept_multiple_files=False
)

if uploaded_file:
    # Load the single document
    docs = load_uploaded_files([uploaded_file])
    full_text = "\n".join([doc.page_content for doc in docs])

    if action == "Summarize Document":
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
        st.subheader("🔍 Ask a Question")
        question = st.text_input("Enter your question about the document:")
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

            st.write(answer)
            st.subheader("📚 Sources")
            with st.expander("Show sources"):
                for src in sources:
                    st.write(src.metadata.get("source"))
        else:
            st.info("Please enter a question to get an answer.")
else:
    st.info("Upload a document to get started.")
