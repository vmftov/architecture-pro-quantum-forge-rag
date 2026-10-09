from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from textencoder import TextEncoder

# Верхняя граница размера чанка
# Судя по статьям, для большинства retrieval-задач чанки > 1024 токенов редко дают выигрыш
CHUNK_SIZE_QUALITY_CAP = 1024


class TextSplitter:
    def __init__(self, encoder: TextEncoder, show_stats: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(encoder.get_model_name())
        self.tokenizer.model_max_length = int(1e12)  # отключаем warning для подсчёта

        # оставляем 15-20% на спецтокены и неточности сплиттера
        chunk_size = min(int(encoder.get_max_seq_length_tokens() * 0.8), CHUNK_SIZE_QUALITY_CAP)
        # перекрытие текстов 10-20% от chunk_size
        chunk_overlap = int(chunk_size * 0.2)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.calc_length_tokens,
            add_start_index=True
        )

        if show_stats:
            print(f"Text-chunk splitter stats:")
            print(f"   splitter type: RecursiveCharacterTextSplitter")
            print(f"   chunk_size: {chunk_size} (symbols)")
            print(f"   chunk_overlap: {chunk_overlap} (symbols)")
            print(f"")

    def calc_length_tokens(self, text: str) -> int:
        """Возвращает длину текста в токенах"""
        # без add_special_tokens=False добавляет [CLS], [SEP]
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def split_texts(self, texts: list[str]) -> list[Any]:
        """Разбивает массив текстов на чанки и возвращает объекты документов"""
        return self.splitter.create_documents(texts)
