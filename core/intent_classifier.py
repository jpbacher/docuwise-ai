import os
from typing import Literal
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

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

