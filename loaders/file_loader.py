import os
import streamlit as st
from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
import tempfile


def load_uploaded_files(uploaded_files: List) -> List[Document]:
    """
    Loads and parses uploaded PDF, TXT, and CSV files into LangChain Document 
    objects, overriding each document's source metadata 
    with the original filename.

    Parameters:
    -----------
    uploaded_files : List
        A list of Streamlit UploadedFile objects.

    Returns:
    --------
    List[Document]
        Parsed documents ready for chunking and embedding, with readable source metadata.
    """
    documents: List[Document] = []

    for uploaded in uploaded_files:
        # Write the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded.read())
            tmp_path = tmp_file.name

        try:
            # Select appropriate loader
            if uploaded.name.lower().endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            elif uploaded.name.lower().endswith(".csv"):
                loader = CSVLoader(tmp_path)
            elif uploaded.name.lower().endswith(".txt"):
                loader = TextLoader(tmp_path)
            else:
                st.warning(f"Unsupported file type: {uploaded.name}")
                continue

            # Load documents and override source metadata
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = uploaded.name
            documents.extend(docs)

        finally:
            # Clean up temp file
            os.remove(tmp_path)

    return documents
