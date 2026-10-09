from dataclasses import dataclass
from pathlib import Path


@dataclass
class ChunkData:
    start_index: int
    end_index: int
    doc_index: int
    saved_text: str


class DocumentInMemoryStorage:
    def __init__(self):
        self.doc_paths: list[Path] = []
        self.chunk_data_list: list[ChunkData] = []

    def append_document(self, file_path: Path) -> int:
        """Добавляет документ и возвращает его индекс"""
        index = len(self.doc_paths)
        self.doc_paths.append(file_path)
        return index

    def get_document_path(self, doc_index: int) -> str:
        return self.doc_paths[doc_index]

    def append_chunks(self, chunks: list[ChunkData]):
        """Добавляет чанки"""
        self.chunk_data_list.extend(chunks)

    def get_chunk_data(self, index: int):
        return self.chunk_data_list[index]
