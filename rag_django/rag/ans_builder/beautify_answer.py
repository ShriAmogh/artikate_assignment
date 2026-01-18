from typing import Dict, List
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
#print(os.getenv("GEMINI_API_KEY"))
#print("==========================================")


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
        self, question: str, top_documents: List[Dict]
    ) -> Dict:

        context = self._merge_context(top_documents)

        sources = self._collect_sources(top_documents)

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
            "sources": sources,
        }

    def _merge_context(self, documents: List[Dict]) -> str:
        """
        Merge multiple chunks into a single clean context
        """
        merged = []

        for doc in documents:
            content = doc.get("content", "")
            if content:
                merged.append(self._normalize_context(content))

        return "\n\n".join(merged)

    def _normalize_context(self, text: str) -> str:
        """
        Normalize bullets and spacing for LLM consumption
        """
        return (
            text.replace("•", "-")
                .replace("\n\n", "\n")
                .strip()
        )

    def _collect_sources(self, documents: List[Dict]) -> List[str]:
        """
        Deduplicate and format sources
        """
        seen = set()
        sources = []

        for doc in documents:
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown Document")
            page = metadata.get("page", "N/A")

            key = f"{source}-{page}"
            if key not in seen:
                seen.add(key)
                sources.append(f"{source} - Page {page}")

        return sources


    def _build_prompt(self, question: str, context: str) -> str:
        return f"""
You are a knowledge assistant answer from the provided textbook excerpt.

TASK:
- Extract the answer stated in the context.
- Preserve lists, bullet points, and classifications.
- If the context contains a classification, output all items completely.
- Do NOT truncate sentences.

CONTEXT:
\"\"\"
{context}
\"\"\"

QUESTION:
{question}

IMPORTANT:
-Always Finish the answer completely based on the CONTEXT provided.

INSTRUCTIONS:
- If the answer is a list, reproduce the list fully.
- Always base your answer on the provided context.
- Always answer in the third person.

FINAL ANSWER:
"""
