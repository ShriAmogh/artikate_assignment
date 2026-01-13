from typing import Dict, List
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


class AnswerBeautifier:


    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 512,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def generate_answer(
        self, question: str, top_document: Dict
    ) -> Dict:

        context = top_document["content"]
        metadata = top_document.get("metadata", {})

        prompt = self._build_prompt(question, context)

        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )

        return {
            "answer": response.text.strip(), 
            "sources": self._format_source(metadata),
        }

    def _build_prompt(self, question: str, context: str) -> str:

        return f"""
You are a strict knowledge assistant.

RULES:
- Answer ONLY using the provided context.
- Do NOT add any external knowledge.
- If the answer is not present, say:
  "The information is not available in the provided document."


CONTEXT:
\"\"\"
{context}
\"\"\"

QUESTION:
{question}

ANSWER:
"""

    def _format_source(self, metadata: Dict) -> List[str]:
        source = metadata.get("source", "Unknown Document")
        page = metadata.get("page", "N/A")
        return [f"{source} - Page {page}"]
