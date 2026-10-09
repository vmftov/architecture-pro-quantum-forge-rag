from dataclasses import dataclass
from datetime import datetime
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import logging


def create_s3_client(endpoint_url: str, region_name: str, access_key_id: str, secret_access_key: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region_name,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        config=Config(s3={"addressing_style": "path"})
    )


@dataclass
class S3File:
    key: str
    etag: str
    modifiedat: datetime


class S3Bucket:
    def __init__(self, client, bucket_name: str, region_name: str):
        self.bucket_name = bucket_name
        self.region_name = region_name
        self.client = client
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            logger = logging.getLogger(__name__)
            error_code = e.response["Error"]["Code"]
            if error_code == "403":
                logger.error(f"Доступ к бакету '{self.bucket_name}' запрещён")
                raise
            elif error_code == "404":
                logger.debug(f"Создание бакета '{self.bucket_name}'")
                self._create_bucket()
                logger.info(f"Бакет '{self.bucket_name}' создан")
            else:
                raise

    def _create_bucket(self):
        try:
            self.client.create_bucket(
                Bucket=self.bucket_name,
                CreateBucketConfiguration={"LocationConstraint": self.region_name},
            )
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                raise

    def list_all_files(self, prefix="") -> list[S3File]:
        paginator = self.client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

        files: list[tuple[str, str]] = []
        for page in pages:
            for obj in page.get('Contents', []):
                files.append(S3File(key=obj['Key'], etag=obj['ETag'], modifiedat=obj['LastModified']))
        return files

    def try_get_file_text(self, key: str) -> str | None:
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=key)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ('NoSuchKey', '404'):
                return None
            raise

        body = response['Body'].read()
        if isinstance(body, str):
            return body

        return body.decode("utf-8")

    def put_file_text(self, key: str, text: str):
        self.client.put_object(Bucket=self.bucket_name, Key=key, Body=text.encode("utf-8"))

    def delete_file(self, key: str):
        self.client.delete_object(Bucket=self.bucket_name, Key=key)
