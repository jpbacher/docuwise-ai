import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStore

# Load environment variables from .env
load_dotenv()


def get_qa_chain(vectorstore: VectorStore) -> RetrievalQA:
    """
    Create and return a RetrievalQA chain that uses an OpenAI LLM
    and retrieves context from the provided vector store.

    Args:
        vectorstore (VectorStore): A LangChain VectorStore instance
            implementing `as_retriever()` to fetch relevant document chunks.

    Returns:
        RetrievalQA: A QA chain configured with the OpenAI model and
            the vectorstore retriever, ready to answer queries.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
 
    # Initialize the OpenAI chat model
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.0,
        openai_api_key=api_key)

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain
