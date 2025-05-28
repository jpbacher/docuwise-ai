import streamlit as st
import os
from dotenv import load_dotenv
from loaders.file_loader import load_uploaded_files
from core.vector_store import build_vector_store
from core.qa_chain import get_qa_chain



# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="RAG Assistant", layout="centered")
st.title("📄🧠 Document Q&A Assistant")

uploaded_files = st.file_uploader("Upload your documents", type=["pdf", "csv", "txt"], accept_multiple_files=True)
question = st.text_input("Ask a question about your documents:")

if uploaded_files and question:
    docs = load_uploaded_files(uploaded_files)
    vectorstore = build_vector_store(docs)
    qa_chain = get_qa_chain(vectorstore)

    with st.spinner("Generating answer..."):
        response = qa_chain.invoke({"query": question})
        answer = response["result"]
        sources = response["source_documents"]

    st.success(answer)
