from typing import Any

from numpy import dtype, ndarray
from sentence_transformers import SentenceTransformer


class TextEncoder:
    def __init__(self, model_name: str, batch_size: int, show_stats: bool = False):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

        embedding_dim = self.model.get_embedding_dimension()
        max_seq_length = self.model.get_max_seq_length()

        if show_stats:
            print(f"Embedding model stats:")
            print(f"   model name: '{model_name}'")
            print(f"   embedding_dimension: {embedding_dim}")
            print(f"   max_seq_length: {max_seq_length} (tokens)")
            print(f"")

    def get_model_name(self):
        """Возвращает имя модели"""
        return self.model_name

    def get_max_seq_length_tokens(self):
        """Возвращает максимальное кол-во токенов на входе"""
        return self.model.get_max_seq_length()

    def get_embedding_dimension(self):
        """Возвращает размер эмбеддинга (вектора)"""
        return self.model.get_embedding_dimension()

    def encode_text(self, inputs: list[str], show_progress_bar) -> ndarray[Any, dtype[Any]]:
        """Возвращает вектор для текста"""
        return self.model.encode(
            inputs,
            normalize_embeddings=True,                # обязательно для BGE
            batch_size=self.batch_size,               #
            show_progress_bar=show_progress_bar,      #
            convert_to_numpy=True,                    # для векторной БД
        ).astype("float32")                           # для faiss или qdrant
