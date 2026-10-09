from datetime import datetime
import os
import requests
import re
import sys
import unicodedata
import urllib.parse
from pathlib import Path
from dataclasses import dataclass

# Даные документа для сохранения в файл
@dataclass
class DocumentFileInfo:
    file_name: str
    contents: bytes

# Создаёт заданную директорию
def make_dir(dir_name):
    folder_path = Path(dir_name)
    folder_path.mkdir(parents=True, exist_ok=True)

# Корректирует ULR
def fix_url(url):
    if not bool(re.search(r'%[0-9A-Fa-f]{2}', url)):
        parsed_url = urllib.parse.urlparse(url)
        path = urllib.parse.quote(parsed_url.path, safe='/')
        url = urllib.parse.urlunparse((parsed_url.scheme, parsed_url.netloc, path, parsed_url.params, parsed_url.query, parsed_url.fragment))
    return url

# Возвращает путь к документу в форме имени файла
def get_path_as_file_name(url):
    parsed_url = urllib.parse.urlparse(url)
    file_name = urllib.parse.unquote_plus(parsed_url.path)
    file_name = file_name.strip("/")
    file_name = file_name.replace("/", "--")
    file_name = unicodedata.normalize("NFC", file_name)
    file_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", file_name)
    file_name = file_name.rstrip()

    if len(file_name) == 0:
        raise ValueError("Путь к документу пустой")
    
    return file_name

# Считывает содержимое документа по URL
def fetch_doc(url: str):
    try:
        print(f"{datetime.now().isoformat()} - Скачивание документа '{url}'")

        url = fix_url(url)
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')
        file_name = get_path_as_file_name(url)
        
        if "text/html" in content_type:
            return DocumentFileInfo(file_name=f"{file_name}.html", contents=response.content)
        else:
            raise ValueError("Тип данных документа не поддерживается")
    except requests.exceptions.Timeout:
        print(f"При скачивании документа произошла ошибка: Истек таймаут ожидания ответа.", file=sys.stderr)
    except Exception as err:
        print(f"При скачивании документа произошла ошибка: {err}.", file=sys.stderr)

    return None

# Сохраняет документ в файл
def save_doc_to_file(folder: str, doc: DocumentFileInfo):
    try:
        file_path = os.path.join(folder, doc.file_name)
    
        print(f"{datetime.now().isoformat()} - Сохранение документа '{doc.file_name}' в файл '{file_path}'")
    
        with open(file_path, "wb") as f:
            f.write(doc.contents)
    except Exception as err:
        print(f"При сохранении документа произошла ошибка: {err}.", file=sys.stderr)

# Считывает документы из файла-списка download-list и загружает их в папку docs по группам
def fetch_docs(docs_list_file_path: str, result_fold_path: str):
    cur_folder = os.path.join(result_fold_path, "other")
    make_dir(cur_folder)

    with open(docs_list_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            cline = line.strip()
            if len(cline) == 0:
                continue

            if cline.startswith("#"):
                group_name = cline[1:].strip()
                if len(group_name) > 0:
                    cur_folder = os.path.join(result_fold_path, group_name)
                    make_dir(cur_folder)
                continue

            doc_info = fetch_doc(cline)
            if doc_info is not None:
                save_doc_to_file(cur_folder, doc_info)

# Скачиваем документы
if __name__ == "__main__":
    docs_list_file_path = arg1 = sys.argv[1]
    result_fold_path = sys.argv[2]
    fetch_docs(docs_list_file_path, result_fold_path)
