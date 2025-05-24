from typing import List
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
import tempfile
import os
import streamlit as st

def load_uploaded_files(uploaded_files: List) -> List[Document]:
    """
    Loads and parses uploaded PDF, TXT, and CSV files into LangChain Document objects.

    Parameters:
    -----------
    uploaded_files : List
        A list of Streamlit UploadedFile objects.

    Returns:
    --------
    List[Document]
        Parsed documents ready for chunking and embedding.
    """
    documents = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            file_path = tmp_file.name

        try:
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file.name.endswith(".csv"):
                loader = CSVLoader(file_path)
            elif file.name.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue

            documents.extend(loader.load())

        finally:
            os.remove(file_path)

    return documents
