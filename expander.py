from openai import OpenAI

_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

_PROMPT = """\
Rewrite the question below using full terminology, expanded acronyms, and relevant synonyms.
Return only the rewritten question as one sentence. No explanation, no preamble.

Question: {question}"""


def expand_query(question: str) -> str:
    try:
        response = _client.chat.completions.create(
            model="llama3.2",
            temperature=0,
            messages=[{"role": "user", "content": _PROMPT.format(question=question)}],
        )
        expanded = response.choices[0].message.content.strip()
        return expanded if expanded else question
    except Exception:
        return question
