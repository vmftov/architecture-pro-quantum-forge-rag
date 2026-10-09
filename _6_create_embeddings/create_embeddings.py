from datetime import datetime, timezone
import os
import sys
from pathlib import Path
from typing import Any
import uuid

# def save_metadata(file_path: str, docs: list[Any], embedding_ids: list[str]):
#     doc_index = doc_stor.append_document(file_path)
#     doc_chunks: list[ChunkData] = []

#     for d in docs:

#         doc_chunks.append(ChunkData(start_index=start_index, end_index=end_index, doc_index=doc_index, saved_text=d.page_content))

#     doc_stor.append_chunks(doc_chunks)


def show_doc_stats(file_path: str, file_content: str, docs: list[Any], embeddings: Any):
    token_counts = [splitter.calc_length_tokens(d.page_content) for d in docs]
    total_tokens = splitter.calc_length_tokens(file_content)

    print("------------------------------------------------------------------------------")
    print(f"Файл: {file_path.name}")
    # print(f"  Содержимое [0]: {docs[0].page_content}")
    # print(f"  Метаданные [0]: {docs[0].metadata}")
    print(f"  Символов: {len(file_content)}")
    print(f"  Токенов всего: {total_tokens}")
    if total_tokens:
        print(f"  Символов в токене: {len(file_content) / total_tokens}")
    print(f"  Чанков: {len(docs)}")
    if token_counts:
        print(
            f"  Токенов в чанке: min={min(token_counts)}, max={max(token_counts)}, avg={sum(token_counts)//len(token_counts)}")
    print(f"  Кол-во эмбеддингов: {len(embeddings)}")
    print(f"  Форма массива эмбеддингов: {embeddings.shape}")
    # print(f"  Первый эмбеддинг: {embeddings[0]}")


def process_docs(docs_dir: str):
    for file_path in Path(docs_dir).rglob("*.anon.md"):
        if file_path.is_file():
            try:
                file_content = file_path.read_text(encoding="utf-8", errors="replace")

                docs = splitter.split_texts([file_content])
                if not docs:
                    print(f"WARN: Splitter return nothing for file '{str(file_path)}'")
                    continue

                embeddings = encoder.encode_text([d.page_content for d in docs], show_progress_bar=True)

                for doc, emb in zip(docs, embeddings):
                    metadata = EmbeddingMetadata(
                        id=str(uuid.uuid4()),
                        doc_id=str(uuid.uuid4()),
                        doc_name=file_path.name,
                        category=file_path.parent.name,
                        language="en-US",
                        modified_at=datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc),
                        start_index=doc.metadata["start_index"],
                        end_index=doc.metadata["start_index"] + len(doc.page_content),
                        text=doc.page_content
                    )
                    searcher.append_single_embedding(emb, metadata)
                    # save_metadata(file_path, docs, embedding_ids)

                show_doc_stats(file_path, file_content, docs, embeddings)
            except Exception as err:
                print(f"При обработке файла произошла ошибка: {err}", file=sys.stderr)
                raise


if __name__ == "__main__":
    os.environ['HF_HOME'] = "G:\\Edu\\_sprint_7\\_models"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from _6_create_embeddings.text_encoder import TextEncoder
    from _6_create_embeddings.text_splitter import TextSplitter
    from _6_create_embeddings.qdrant_semantic_searcher import QdrantSemanticSearcher, EmbeddingMetadata

    model_name = "BAAI/bge-base-en-v1.5"
    # model_name = "BAAI/bge-m3"
    model_batch_size = 32

    encoder = TextEncoder(model_name, model_batch_size, show_stats=True)
    splitter = TextSplitter(encoder, show_stats=True)

    searcher = QdrantSemanticSearcher("http://localhost:6333", "docs")
    collection_exists = searcher.collection_exists()
    searcher.init_collection(encoder)

    if not collection_exists:
        process_docs("G:\\Edu\\_sprint_7\\_work_data\\res")
