import os
from typing import List
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


# Load environment variables from .env
load_dotenv()


def build_vector_store(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> Chroma:
    """
    Build a vector store from a list of documents.

    Args:
        documents (List[Document]): List of documents to be indexed.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        Chroma: The vector store instance.
    """
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Create and return the vector store
    return Chroma.from_documents(texts, embeddings)
