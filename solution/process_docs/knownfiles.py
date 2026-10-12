from dataclasses import asdict, dataclass
import json


@dataclass
class KnownFileInfo:
    # Ключ файла с оригинальным содержимым
    asis_key: str
    # Ключ файла с анонимизированным содержимым
    anon_key: str
    # Ключ оригинального файла
    orig_key: str
    # ETag оригинального файла
    orig_etag: str


class KnownFilesMetadata:
    def __init__(self):
        self.files: list[KnownFileInfo] = []

    def from_string(self, contents: str | None):
        if not contents:
            return
        data = json.loads(contents)
        self.files = [KnownFileInfo(**item) for item in data]

    def to_string(self) -> str:
        data = [asdict(item) for item in self.files]
        return json.dumps(data, ensure_ascii=False, indent=4)

    def add_file(self, file: KnownFileInfo):
        self.files.append(file)

    def remove_file(self, file: KnownFileInfo):
        self.files.remove(file)
