import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = OpenAI(
    base_url=os.environ["OLLAMA_BASE_URL"],
    api_key="ollama",
)

_PROMPT = """\
Rewrite the question below using full terminology, expanded acronyms, and relevant synonyms.
Return only the rewritten question as one sentence. No explanation, no preamble.

Question: {question}"""


def expand_query(question: str) -> str:
    try:
        response = _client.chat.completions.create(
            model=os.environ["OLLAMA_MODEL"],
            temperature=0,
            messages=[{"role": "user", "content": _PROMPT.format(question=question)}],
        )
        expanded = response.choices[0].message.content.strip()
        return expanded if expanded else question
    except Exception:
        return question
