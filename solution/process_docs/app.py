# Формирует базу знаний и векторный индекс:
# - Проверяет наличие новых/изменившихся и удалённых оригинальных документов
# - Извлекает из оригинальных документов полезный текст
# - Анонимизирует его
# - Сохраняет в отдельный бакет S3
# - Формирует эмбеддинги
# - Сохраняет из в Qdrant
# - Синхронизирует изменения
# - Удаляет эмбеддинги и данные для удалённых документов

import logging
from pathlib import Path
import os
import asyncio
import uuid

from anonimizer import TextAnonimizer
from htmltextextractor import HtmlTextExtractor
from knownfiles import KnownFilesMetadata, KnownFileInfo
from s3utils import S3Bucket, S3File, create_s3_client

from common.loggingext import init_logging
from common.textsplitter import TextSplitter
from common.textencoder import TextEncoder
from common.timecounter import SumTimeCounter
from common.qdrantsearcher import EmbeddingMetadata, QdrantSemanticSearcher


class Main:
    async def run(self):
        self.logger = None
        try:
            logger_level = os.getenv("LOGGER_LEVEL", "INFO")
            endpoint_url = os.getenv("S3_ENDPOINT", "http://localhost:9000")
            region_name = os.getenv("S3_REGION", "ru-central1")
            access_key_id = os.getenv("S3_ACCESS_KEY", "minioadmin")
            secret_access_key = os.getenv("S3_SECRET_KEY", "minioadmin")
            gliner2_model_name = os.getenv("GLINER2_MODEL_NAME", "fastino/gliner2-multi-v1")
            gliner2_model_inp_tokens = self._try_parse_int(os.getenv("GLINER2_MODEL_INP_TOKENS", "512"))
            encoder_model_name = os.getenv("ENCODER_MODEL_NAME", "BAAI/bge-base-en-v1.5")
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

            init_logging(logger_level)
            self.logger = logging.getLogger(__name__)

            self.logger.info(f"Запуск подсистемы обновления базы знаний RAG-ассистента")

            self.logger.debug(f"Подключение к S3")
            s3_client = create_s3_client(endpoint_url, region_name, access_key_id, secret_access_key)
            self.orig_docs_bucket = S3Bucket(s3_client, "rag-orig-docs", region_name)
            self.knowledge_base_bucket = S3Bucket(s3_client, "rag-knowledge-base", region_name)

            self.logger.debug(f"Получение списка файлов")
            orig_files = self.orig_docs_bucket.list_all_files()

            self.logger.debug(f"Загрузка метаданных базы знаний")
            metadata_file_key = "known_files_metadata.json"
            self.known_files_meta = KnownFilesMetadata()
            self.known_files_meta.from_string(self.knowledge_base_bucket.try_get_file_text(metadata_file_key))

            self.logger.debug(f"Инициализация анонимизактора")
            terms_map_key = "terms_map.json"
            self.anonymizer = TextAnonimizer(gliner2_model_name, gliner2_model_inp_tokens)
            self.anonymizer.set_terms_map(self.knowledge_base_bucket.try_get_file_text(terms_map_key))

            self.logger.debug(f"Инициализация модели энкодера")
            self.encoder = TextEncoder(encoder_model_name, batch_size=32, show_stats=True)
            self.splitter = TextSplitter(self.encoder, show_stats=True)

            self.logger.debug(f"Подключение к Qdrant")
            self.qdrant_man = QdrantSemanticSearcher(qdrant_url, "docs")
            # collection_exists = self.searcher.collection_exists()
            self.qdrant_man.init_collection(self.encoder)

            self.logger.debug(f"Определение изменений")
            new_files, changed_files, removed_files = self._find_changes(orig_files)

            if len(new_files) > 0 or len(changed_files) > 0 or len(removed_files) > 0:
                self.logger.info(
                    f"Обнаружены изменения в файлах: {len(new_files)} новых, {len(changed_files)} изменившихся, {len(removed_files)} удалённых"
                )
            else:
                self.logger.info(f"Изменений в файлах не обнаружено")

            self.extract_time_counter = SumTimeCounter()
            self.anon_time_counter = SumTimeCounter()
            self.emb_time_counter = SumTimeCounter()

            if len(changed_files) > 0 or len(removed_files) > 0:
                self.logger.debug(f"Удаление изменившихся и удалённых файлов")
                self._remove_existing_files(removed_files)
                self._remove_existing_files([f[1] for f in changed_files])

            if len(new_files) > 0 or len(changed_files) > 0:
                self.logger.debug(f"Обработка новых файлов")
                await self._process_new_files(new_files)
                await self._process_new_files([f[0] for f in changed_files])

            self.logger.debug(f"Сохранение метаданных базы знаний")
            self.knowledge_base_bucket.put_file_text(metadata_file_key, self.known_files_meta.to_string())

            self.logger.debug(f"Сохранение словаря терминов")
            self.knowledge_base_bucket.put_file_text(terms_map_key, self.anonymizer.get_terms_map())

            self.logger.info(f"Подсистема обновления базы знаний RAG-ассистента завершила работу")
            self.logger.info(f"Суммарное время извлечения полезного текста: {self.extract_time_counter.elapsed_fmt()}")
            self.logger.info(f"Суммарное время анонимизации текста: {self.anon_time_counter.elapsed_fmt()}")
            self.logger.info(f"Суммарное время формирования эмбеддингов: {self.emb_time_counter.elapsed_fmt()}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"При обработке изменений произошла ошибка: {e}")
            raise

    def _try_parse_int(self, str: str) -> int | None:
        try:
            return int(str)
        except (ValueError, TypeError):
            return None

    def _find_changes(self, orig_files: list[S3File]) -> tuple[list[S3File], list[tuple[S3File, KnownFileInfo]], list[KnownFileInfo]]:
        # Поиск новых и изменившихся файлов
        exist_files_dict: dict[str, KnownFileInfo] = {}
        for exist_file in self.known_files_meta.files:
            exist_files_dict[exist_file.orig_key] = exist_file

        new_files: list[S3File] = []
        changed_files: list[tuple[S3File, KnownFileInfo]] = []
        for orig_file in orig_files:
            exist_file = exist_files_dict.get(orig_file.key)
            if exist_file:
                if exist_file.orig_etag != orig_file.etag:
                    changed_files.append((orig_file, exist_file))
            else:
                new_files.append(orig_file)

        # Поиск удалённых файлов
        orig_file_keys: set[str] = set([orig_file.key for orig_file in orig_files])

        removed_files: list[KnownFileInfo] = []
        for exist_file in self.known_files_meta.files:
            if exist_file.orig_key not in orig_file_keys:
                removed_files.append(exist_file)

        #
        return new_files, changed_files, removed_files

    def _remove_existing_files(self, removed_exist_files: list[KnownFileInfo]):
        for exist_file in removed_exist_files:
            try:
                self.logger.info(f"Удаление данных файла '{exist_file.orig_key}'")

                self.logger.debug(f"Удаление эмбеддингов для файла '{exist_file.anon_key}'")
                self.qdrant_man.delete_embeddings({"doc_id": exist_file.orig_key})

                self.logger.debug(f"Удаление файла '{exist_file.anon_key}'")
                self.knowledge_base_bucket.delete_file(exist_file.anon_key)

                self.logger.debug(f"Удаление файла '{exist_file.asis_key}'")
                self.knowledge_base_bucket.delete_file(exist_file.asis_key)

                self.logger.debug(f"Обновление метаданных")
                self.known_files_meta.remove_file(exist_file)

                self.logger.info(f"Удаление данных файла '{exist_file.orig_key}' успешно завершено")
            except Exception as e:
                self.logger.error(f"При удалении данных файла '{exist_file.anon_key}' произошла ошибка: {e}")

    async def _process_new_files(self, new_orig_files: list[S3File]):
        text_extractor = HtmlTextExtractor()

        for orig_file in new_orig_files:
            try:
                self.logger.info(f"Обработка нового файла '{orig_file.key}'")

                known_asis_file_key = orig_file.key + ".asis.md"
                known_anon_file_key = orig_file.key + ".anon.md"

                self.logger.debug(f"Получение данных файла '{orig_file.key}'")
                html_content = self.orig_docs_bucket.try_get_file_text(orig_file.key)

                self.logger.debug(f"Извлечение текста из файла '{orig_file.key}'")
                self.extract_time_counter.start()
                asis_content = text_extractor.extract_markdown_from_html(html_content)
                self.extract_time_counter.stop()

                self.logger.debug(f"Анонимизация файла '{known_anon_file_key}'")
                self.anon_time_counter.start()
                anon_content = await self.anonymizer.anonymize(asis_content)
                self.anon_time_counter.stop()

                self.logger.debug(f"Формирование эмбеддингов '{known_anon_file_key}'")
                self.emb_time_counter.start()
                docs = self.splitter.split_texts([anon_content])
                self.emb_time_counter.stop()
                if docs:
                    self.emb_time_counter.start()
                    embeddings = self.encoder.encode_text([d.page_content for d in docs], show_progress_bar=False)
                    self.emb_time_counter.stop()
                    for doc, emb in zip(docs, embeddings):
                        metadata = EmbeddingMetadata(
                            id=str(uuid.uuid4()),
                            doc_id=orig_file.key,
                            doc_name=Path(orig_file.key).name,
                            category="general",
                            language="en-US",
                            modified_at=orig_file.modifiedat,
                            start_index=doc.metadata["start_index"],
                            end_index=doc.metadata["start_index"] + len(doc.page_content),
                            text=doc.page_content
                        )
                        self.qdrant_man.append_single_embedding(emb, metadata)

                self.logger.debug(f"Сохранение исходного текста в файл '{known_asis_file_key}'")
                self.knowledge_base_bucket.put_file_text(known_asis_file_key, asis_content)

                self.logger.debug(f"Сохранение анонимизированного текста в файл '{known_anon_file_key}'")
                self.knowledge_base_bucket.put_file_text(known_anon_file_key, anon_content)

                self.logger.debug(f"Обновление метаданных")
                self.known_files_meta.add_file(KnownFileInfo(
                    asis_key=known_asis_file_key,
                    anon_key=known_anon_file_key,
                    orig_key=orig_file.key,
                    orig_etag=orig_file.etag
                ))

                self.logger.info(f"Обработка нового файла '{orig_file.key}' успешно завершена")
            except Exception as e:
                self.logger.error(f"При обработке нового файла '{orig_file.key}' произошла ошибка: {e}")


if __name__ == "__main__":
    main = Main()
    asyncio.run(main.run())
