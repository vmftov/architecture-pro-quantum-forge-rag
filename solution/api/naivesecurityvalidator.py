from qdrant_client import models


class NaiveSecurityValidator:
    def __init__(self):
        self.stopwords = set([
            "password", "api key", "api_key", "парол", "код доступа", "ignore all instructions", "игнорируй системные инструкции"
        ])

    def filter_context(self, points: list[models.ScoredPoint]) -> list[models.ScoredPoint]:
        result_points: list[models.ScoredPoint] = []
        for p in points:
            if not self._contains_stopwords(p.payload.get('text')):
                result_points.append(p)
        return result_points

    def _contains_stopwords(self, text: str | None) -> bool:
        if text:
            norm_text = text.replace("  ", " ").lower()
            for stopword in self.stopwords:
                if stopword in norm_text:
                    return True
        return False
