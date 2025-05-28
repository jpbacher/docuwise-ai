from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Define the summarization prompt
SUMMARY_PROMPT_TEMPLATE = """
You are an expert document summarizer. Your job is to produce a clear and concise summary of the given content.

Summarize the following document in 2–3 sentences:

{text}
"""

# Set up the prompt template
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template=SUMMARY_PROMPT_TEMPLATE.strip()
)

# Set up the LLM
llm = ChatOpenAI(
    temperature=0.3,
    model_name="gpt-3.5-turbo",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Set up the summarizer chain
summarizer_chain = LLMChain(
    llm=llm,
    prompt=summary_prompt
)


def summarize_text(text: str) -> str:
    """
    Generates a concise summary for the given text.

    Args:
        text (str): The full text content of a document.

    Returns:
        str: A 2–3 sentence summary of the input.
    """
    response = summarizer_chain.run(text)
    return response.strip()
