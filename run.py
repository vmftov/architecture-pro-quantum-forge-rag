from pathlib import Path
import os
import subprocess
import sys

working_dir = str(Path.cwd())
docs_list_file_path = os.path.join(working_dir, "doc-list.txt")
docs_fold_path = os.path.join(working_dir, "_work_data", "docs")
ner_folder_path = os.path.join(working_dir, "_work_data", "ner")
models_fold_path = os.path.join(working_dir, "_models")

def run_script(title: str, script_folder: str, script_file: str, *args: str):
    print(f"\n# {title}\n")
    script_path = os.path.join(working_dir, script_folder, script_file)
    result = subprocess.run([sys.executable, script_path, *args])
    if result.returncode != 0:
        raise ValueError(f"Скрипт {script_file} завершился с ошибкой, код {result.returncode}")

# Последовательно запускает действия по созданию семантического поиска 
def run_pipeline():
    try:
        #run_script("Скачивание файлов", "_1_fetch_docs", "fetch_docs.py", docs_list_file_path, docs_fold_path)
        #run_script("Извлечение текста", "_2_extract_text", "extract_text.py", docs_fold_path)
        #run_script("Анонимизация текста", "_3_anonymize", "extract_ner.py", docs_fold_path, models_fold_path, ner_folder_path)
        #run_script("Перевод текста", "_4_translate_ru", "translate_ru.py", docs_fold_path, models_fold_path)
        #run_script("Копирование", "_5_copy", "copy.py")
        run_script("Создание эмбеддингов и прочее", "_6_create_embeddings", "create_embeddings.py")
    except Exception as err:
        print(f"Пайплайн завершился с ошибкой: {err}", file=sys.stderr)

if __name__ == "__main__":
    run_pipeline()
