from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

from qdrant_client import QdrantClient, models
from textencoder import TextEncoder


@dataclass
class EmbeddingMetadata:
    id: str
    doc_id: str
    doc_name: str
    category: str
    language: str
    modified_at: datetime
    start_index: int
    end_index: int
    text: str


class QdrantSemanticSearcher:
    def __init__(self, url: str, collection_name: str):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def collection_exists(self):
        return self.client.collection_exists(self.collection_name)

    def init_collection(self, encoder: TextEncoder):
        self._ensure_collection(self.collection_name, encoder)
        self._validate_collection(self.collection_name, encoder)

    def _ensure_collection(self, collection_name: str, encoder: TextEncoder):
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=encoder.get_embedding_dimension(),
                    distance=models.Distance.COSINE,
                    on_disk=True                        # оригинальные векторы хранятся на диске
                ),
                hnsw_config=models.HnswConfigDiff(
                    on_disk=False,                      # HNSW-граф в RAM — быстрый поиск
                    m=16,                               # рёбер на узел (16–32 — баланс)
                    ef_construct=100,                   # качество построения графа
                ),
                on_disk_payload=True,                   # payload на диске для экономии RAM
            )

    def _validate_collection(self, collection_name: str, encoder: TextEncoder):
        embedding_dimension = encoder.get_embedding_dimension()
        if self.client.collection_exists(collection_name):
            info = self.client.get_collection(collection_name)
            vector_size = info.config.params.vectors.size
            if vector_size != embedding_dimension:
                raise ValueError(
                    f"Коллекция '{collection_name}' существует, но имеет vectors.size={vector_size}, тогда как embedding_dimension={embedding_dimension}.")

    def append_single_embedding(self, embedding: Any, metadata: EmbeddingMetadata):
        """Добавляет эмбеддинг в индекс"""

        payload = asdict(metadata)
        payload["modified_at"] = metadata.modified_at.isoformat()
        points = [{"id": metadata.id, "vector": embedding, "payload": payload}]

        res = self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        if res.status != models.UpdateStatus.COMPLETED:
            raise ValueError(f"Не удалось добавить embedding (статус={res.status})")

    def delete_embeddings(self, payload_filter: dict[str, str]):
        """Удаляет эмбеддинги с заданными метаданными из индекса"""

        if not payload_filter or len(payload_filter) == 0:
            raise ValueError("payload_filter не может быть пустым, т.к. это приведёт к удалению всех векторов в коллекции")

        conditions = []
        for key, value in payload_filter.items():
            conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))

        delete_filter = models.Filter(must=conditions)

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=delete_filter)
        )

    def search(self, embeddings, limit: int = 3) -> list[models.ScoredPoint]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=embeddings[0],
            limit=limit,
            with_payload=True,
            # query_filter={"must": [{"key": "author", "match": {"value": "Alice"}}]}
        )
        return response.points
