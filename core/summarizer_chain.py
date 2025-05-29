import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from prompts.summarization_prompt import SUMMARIZATION_PROMPT

# Load environment variables
load_dotenv()


def get_summarizer_chain() -> LLMChain:
    """
    Creates and returns an LLMChain for summarization using 
    the predefined SUMMARIZATION_PROMPT.
    """
    llm = ChatOpenAI(
        temperature=0.3,
        model_name="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    return LLMChain(llm=llm, prompt=SUMMARIZATION_PROMPT)
