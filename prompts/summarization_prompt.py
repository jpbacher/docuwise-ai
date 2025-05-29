from langchain.prompts import PromptTemplate

SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables=["context", "tone"],
    template="""
You are an expert technical summarizer with a clear and
concise communication style.
Your task is to read the following content and produce a
human-like summary written in a **{tone}** tone.
Use bullet points to make the summary easier to read.
Keep each bullet point concise and focused on the key information,
with a maximum of 1 to 2 sentences per point.
Summarize as thoroughly or briefly as needed — choose the number of points
that best captures the content clearly and efficiently.
Do not include greetings, signature blocks, or unrelated metadata.

Content:
{context}

Your summary:
""".strip()
)
