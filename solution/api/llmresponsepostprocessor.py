import re

from qdrant_client import models


class LlmResponsePostProcessor:
    def __init__(self):
        ...

    def get_source_refs(self, text: str) -> list[str]:
        return re.findall(r"\[([^\]]+)\]", text)

    def process_response(self, text: str, points: list[models.ScoredPoint]) -> str:
        if "заблокирован политикой безопасности" in text.lower():
            return text
        text = self._substitute_doc_names(text, points)
        return text

    def _substitute_doc_names(self, text: str, points: list[models.ScoredPoint]) -> str:
        res_text = text

        # Подставляем документы
        for i, point in enumerate(points):
            doc_name = point.payload.get("doc_name") if point.payload else None
            if doc_name:
                res_text = res_text.replace(f"[{i + 1}]", f"[{doc_name}]")

        # Удаляем повторы
        res_text = re.sub(r"(\[[^\]]+\])\1+", r"\1", res_text)

        return res_text
