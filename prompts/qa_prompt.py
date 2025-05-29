from langchain.prompts import PromptTemplate

# QA_PROMPT: uses context and question (plus tone for styling) to generate answers
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question", "tone"],
    template="""
You are an expert technical assistant with a clear and concise communication style.
Answer the following question in a **{tone}** tone, based only on the context provided.
If the answer cannot be found in the context, respond with "I don't know" or 
"This information is not included in the content."

Context:
{context}

Question:
{question}

Answer:
""".strip()
)
