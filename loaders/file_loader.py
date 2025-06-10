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
        Parsed documents ready for chunking and embedding, with readable
        source metadata.
    """
    documents: List[Document] = []

    for uploaded in uploaded_files:
        # Write the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded.read())
            tmp_path = tmp_file.name

        try:
            fname = uploaded.name.strip().lower()
            if fname.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)

            elif fname.endswith(".csv"):
                # If you know CSVs might have non-UTF8 text:
                loader = CSVLoader(tmp_path, encoding="latin-1")

            elif fname.endswith(".txt"):
                # Either ignore errors or assume latin-1
                loader = TextLoader(tmp_path, encoding="latin-1")
                # Or: TextLoader(tmp_path, encoding="utf-8", errors="ignore")

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
