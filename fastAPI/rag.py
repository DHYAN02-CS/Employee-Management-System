from __future__ import annotations

import math
import os
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from openai import OpenAI


@dataclass(frozen=True)
class EmployeeDocument:
    employee_id: int
    text: str
    embedding: list[float]


class EmployeeRAG:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. Set it before using the RAG assistant."
            )

        self.client = OpenAI(api_key=api_key)
        self.embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self._documents: dict[int, EmployeeDocument] = {}

    def clear(self) -> None:
        self._documents.clear()

    def _document_text(self, employee: Mapping[str, Any]) -> str:
        return (
            f"Employee ID: {employee['emp_id']}\n"
            f"Name: {employee['emp_name']}\n"
            f"Department: {employee['department']}\n"
            f"Salary: {employee['salary']}"
        )

    def _get_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _build_index(self, employees: Sequence[Mapping[str, Any]]) -> None:
        current_ids = {int(employee["emp_id"]) for employee in employees}
        self._documents = {
            employee_id: document
            for employee_id, document in self._documents.items()
            if employee_id in current_ids
        }

        for employee in employees:
            employee_id = int(employee["emp_id"])
            text = self._document_text(employee)
            existing = self._documents.get(employee_id)
            if existing is not None and existing.text == text:
                continue
            self._documents[employee_id] = EmployeeDocument(
                employee_id=employee_id,
                text=text,
                embedding=self._get_embedding(text),
            )

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
            sum(value * value for value in right)
        )
        if denominator == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / denominator

    def answer(
        self, question: str, employees: Sequence[Mapping[str, Any]], top_k: int
    ) -> tuple[str, list[int]]:
        self._build_index(employees)
        if not self._documents:
            return "There are no employee records to answer from.", []

        question_embedding = self._get_embedding(question)
        ranked = sorted(
            self._documents.values(),
            key=lambda document: self._cosine_similarity(
                question_embedding, document.embedding
            ),
            reverse=True,
        )
        sources = ranked[:top_k]
        context = "\n\n".join(document.text for document in sources)

        response = self.client.chat.completions.create(
            model=self.chat_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions about employee records. Use only the "
                        "provided context. If the answer is not in the context, say "
                        "that the employee records do not contain enough information. "
                        "Do not invent employee details or calculations."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        )
        answer = response.choices[0].message.content
        if not answer:
            raise RuntimeError("The OpenAI response did not contain an answer.")
        return answer, [document.employee_id for document in sources]
