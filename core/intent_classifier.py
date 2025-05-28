import os
from dotenv import load_dotenv
from typing import Literal
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


# Load environment variables
load_dotenv()

# Prompt to classify intent
INTENT_PROMPT = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are an assistant that classifies a user's input as either a request
to summarize a document or to ask a question about a document.

Return one word only:
- "summarize" if they want a summary
- "qa" if they are asking a question

User input:
"{user_input}"

Answer:
""".strip()
)


def classify_intent(user_input: str) -> Literal["summarize", "qa"]:
    """Classify user intent as 'summarize' or 'qa'."""
    llm = ChatOpenAI(temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    chain = LLMChain(llm=llm, prompt=INTENT_PROMPT)
    result = chain.run(user_input).strip().lower()

    # Normalize and validate result
    if "sum" in result:
        return "summarize"
    return "qa"
