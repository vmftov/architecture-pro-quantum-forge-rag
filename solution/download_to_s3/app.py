# Скачивает документы из файла download_queue.txt и добавляет их в S3
# Скачанные URL помечаются комментарием

import os
from pathlib import Path
import re
import sys
import unicodedata
import urllib
import requests
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


orig_docs_bucket_name = "rag-orig-docs"


def create_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT", "http://localhost:9000"),
        region_name=os.getenv("S3_REGION", "ru-central1"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY", "minioadmin"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY", "minioadmin"),
        config=Config(s3={"addressing_style": "path"})
    )


def ensure_bucket(s3, bucket_name: str, region: str = "ru-central1") -> bool:
    def create_bucket(s3, bucket_name: str, region: str):
        try:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                raise

    try:
        s3.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        error_code = e.response["Error"]["Code"]

        if error_code == "403":
            print(f"Доступ к бакету '{bucket_name}' запрещён")
            raise
        elif error_code == "404":
            print(f"Создание бакета '{bucket_name}'")
            create_bucket(s3, bucket_name, region)
            print(f"Бакет '{bucket_name}' создан")
            return False
        else:
            raise


def fetch_docs_and_upload_to_s3(docs_list_file_path: Path):
    if docs_list_file_path.is_file() and docs_list_file_path.exists():
        with open(docs_list_file_path, "r", encoding="utf-8") as file:
            for line in file:
                cline = line.strip()
                if len(cline) == 0:
                    continue
                if cline.startswith("#"):
                    pos_group_name = cline[1:].strip()
                    if len(pos_group_name) > 0:
                        group_name = pos_group_name
                    continue
                doc_name, doc_contents = fetch_doc(cline)
                save_doc_to_s3(group_name, doc_name, doc_contents)


def fetch_doc(url: str) -> tuple[str, bytes]:
    try:
        print(f"Скачивание документа '{url}'")

        url = fix_url(url)
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        file_name = get_path_as_file_name(url)

        if "text/html" in content_type:
            content = response.content
            if content:
                return [f"{file_name}.html", content]
            else:
                raise ValueError(f"У документа '{url}' нет содержимого")
        else:
            raise ValueError(f"Тип данных документа '{url}' не поддерживается")
    except requests.exceptions.Timeout:
        print(f"При скачивании документа произошла ошибка: Истек таймаут ожидания ответа.", file=sys.stderr)
    except Exception as err:
        print(f"При скачивании документа произошла ошибка: {err}.", file=sys.stderr)

    return None


def fix_url(url):
    if not bool(re.search(r"%[0-9A-Fa-f]{2}", url)):
        parsed_url = urllib.parse.urlparse(url)
        path = urllib.parse.quote(parsed_url.path, safe="/")
        url = urllib.parse.urlunparse(
            (parsed_url.scheme, parsed_url.netloc, path, parsed_url.params, parsed_url.query, parsed_url.fragment)
        )
    return url


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


def save_doc_to_s3(group_name: str, doc_name: str, doc_contents: bytes):
    try:
        print(f"Сохранение документа '{group_name}/{doc_name}' в S3")

        s3.put_object(
            Bucket=orig_docs_bucket_name,
            Key=f"{group_name}/{doc_name}",
            Body=doc_contents,
            ContentType="text/plain; charset=utf-8",
        )
    except Exception as err:
        print(f"При сохранении документа в S3 произошла ошибка: {err}.", file=sys.stderr)


if __name__ == "__main__":
    s3 = create_s3_client()
    ensure_bucket(s3, orig_docs_bucket_name)
    fetch_docs_and_upload_to_s3(Path("download_queue.txt"))
