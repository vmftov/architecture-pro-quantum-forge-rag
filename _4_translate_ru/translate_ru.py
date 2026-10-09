from dataclasses import dataclass
from datetime import datetime
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _0_common.chunking import split_to_chunks

# Код считывает MD файлы и переводит их содержимое на русский язык
# 
# Чтобы загрузить translator текст разбивается на строки, а строки на чанки
# Чанки подаются в translator в виде единого массива
# По окончании, результат объединяется с помощью маппинга

MAX_LENGTH_TOKENS = 512
MAX_CHUNK_SIZE_SYMBOLS = 400
TRANSLATOR_BATCH_SIZE = 16

# Данные строки
@dataclass
class LineMappingInfo:
    # Индекс строки в оригинальном тексте
    index: int
    # Количество чанков
    chunk_count: int

# Данные датасета
@dataclass
class DataSetInfo:
    # Оригинальные строки
    lines: list[str]
    # Чанки, подающиеся в translator
    dataset: list[str]
    # Маппинг чанков
    mapping: list[LineMappingInfo]

# Создает датасет и маппинг чанков
def create_dataset(text: str) -> DataSetInfo:
    dataset = DataSetInfo(text.splitlines(keepends=False), [], [])
    
    for i in range(len(dataset.lines)):
        if dataset.lines[i].strip():
            chunks, long_chunks = split_to_chunks(dataset.lines[i], MAX_CHUNK_SIZE_SYMBOLS)

            for long_chunk in long_chunks:
                print(f"Не удалось разбить чанк '{long_chunk}' на более мелкие чанки, он будет обрезан.", file=sys.stderr)

            if len(chunks):
                dataset.dataset.extend(chunks)
                dataset.mapping.append(LineMappingInfo(i, len(chunks)))
            
    return dataset

# Объединяет результат перевода в единый текст с учётом маппинга
def join_translation_result(dataset: DataSetInfo, result_lines: list[str]) -> str:
    index = 0
    for m in dataset.mapping:
        chunks = result_lines[index:(index + m.chunk_count)]
        line = " ".join(chunks)
        dataset.lines[m.index] = line
        index += m.chunk_count
    return "\n".join(dataset.lines)

# Выполняет перевод текста
def translate_to_russian(en_content: str, translator: Any) -> str:
    print(f"{datetime.now().isoformat()} -     Подготовка датасета")
    dataset = create_dataset(en_content)
    
    if not len(dataset.dataset):
        print(f"{datetime.now().isoformat()} -     Документ пустой")
        return en_content
    
    print(f"{datetime.now().isoformat()} -     Выполнение перевода")
    result = translator(dataset.dataset, truncation=True, max_length=MAX_LENGTH_TOKENS, batch_size=TRANSLATOR_BATCH_SIZE)
    
    print(f"{datetime.now().isoformat()} -     Обработка результата")
    result_lines = [r["translation_text"] for r in result]
    if len(result_lines) != len(dataset.dataset):
        raise ValueError("Количество результирующих строк меньше исходного числа, переданного в переводчик")
    ru_content = join_translation_result(dataset, result_lines);

    return ru_content

# Сохраняет содержимое в файл
def save_text(file_path: str, text: str):
    print(f"{datetime.now().isoformat()} -     Сохранение файла '{file_path}'")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

# Извлекает полезное содержимое из HTML документов и сохраняет его в файлы        
def translate_documents_to_russian(docs_fold_path: str, translator: Any):
    for file_path in Path(docs_fold_path).rglob("*.md"):
        if file_path.is_file():
            print(f"{datetime.now().isoformat()} - Перевод файла '{file_path}'")
            try:
                en_content = file_path.read_text(encoding="utf-8", errors="replace")
                ru_content = translate_to_russian(en_content, translator)
                save_text(str(file_path.with_suffix(".ru.md")), ru_content)
            except Exception as err:
                print(f"При переводе файла произошла ошибка: {err}", file=sys.stderr)


if __name__ == "__main__":
    docs_fold_path = sys.argv[1]
    models_fold_path = sys.argv[2]

    os.environ['HF_HOME'] = models_fold_path
    from transformers import pipeline
    import torch
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru", device=0)

    print(f"CUDA: {torch.cuda.is_available()}") 
    print(f"Кол-во устройств: {torch.cuda.device_count()}") 
    print(f"Имя устройства: {torch.cuda.get_device_name(0)}")

    translate_documents_to_russian(docs_fold_path, translator)