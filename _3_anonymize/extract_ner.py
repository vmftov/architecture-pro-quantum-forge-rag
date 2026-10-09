from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any
import spacy
from faker import Faker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _0_common.chunking import split_to_chunks

ENTITY_THRESHOLD: float = 0.6
MODEL_MAX_LENGTH_WORDS = 384
MAX_CHUNK_SIZE_SYMBOLS = 2000
MODEL_BATCH_SIZE = 16

# Данные сущности
@dataclass
class EntityInfo:
    # Имя сущности
    text: str
    # Метки
    labels: set[str]
    # 
    fake_text: str

# Данные позиции сущности в документе
@dataclass
class EntityDocPosInfo:
    # Имя сущности
    text: str
    # Позиция
    start: int
    end: int

# Извлекает именованные сущности из текста
def extract_ner(text: str, model: Any, entity_threshold: float, labels: list[str], result_entities: dict[str, EntityInfo]) -> list[EntityDocPosInfo]:
    chunks, long_chunks = split_to_chunks(text, MAX_CHUNK_SIZE_SYMBOLS)
    
    for long_chunk in long_chunks:
        print(f"Не удалось разбить чанк '{long_chunk}' на более мелкие чанки, он будет обрезан.", file=sys.stderr)

    batch_found_entities = model.inference(chunks, labels, threshold=entity_threshold, batch_size=MODEL_BATCH_SIZE)
    entity_positions: list[EntityDocPosInfo] = []

    for found_entities in batch_found_entities:
        for found_entity in found_entities:
            entity_text = found_entity["text"]
            # Сохраняем позицию
            entity_positions.append(EntityDocPosInfo(text=entity_text, start=found_entity["start"], end=found_entity["end"]))
            # Добавляем в общий список
            result_entity = result_entities.get(entity_text)
            if result_entity is None:
                result_entity = EntityInfo(text=entity_text, labels=set(), fake_text="")
                result_entities[entity_text] = result_entity
            result_entity.labels.add(found_entity["label"])

    return entity_positions

# Извлекает именованные сущности из документов
def extract_all_documents_ner(docs_fold_path: str, model: Any, entity_threshold: float) -> dict[str, EntityInfo]:
    labels = ["character", "organization", "planet", "location", "vehicle", "weapon", "machine", "species", "droid", "event", "date", "time"]
    entities: dict[str, EntityInfo] = {}

    for file_path in Path(docs_fold_path).rglob("*.html.md"):
        if file_path.is_file():
            print(f"{datetime.now().isoformat()} - Обработка файла '{file_path}'")
            try:
                file_text = file_path.read_text(encoding="utf-8", errors="replace")
                entity_positions = extract_ner(file_text, model, entity_threshold, labels, entities)
                save_entity_positions(entity_positions, f"{str(file_path)}.pos.json")
            except Exception as err:
                print(f"При обработке файла произошла ошибка: {err}", file=sys.stderr)

    return entities

# Инициализирует модель и выводит статистику
def init_model(model_name: str, models_fold_path: str):
    from gliner import GLiNER
    import torch

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    
    model = GLiNER.from_pretrained(model_name, map_location=device, max_length=MODEL_MAX_LENGTH_WORDS)

    print(f"CUDA: {torch.cuda.is_available()}") 
    print(f"Устройство: {next(model.parameters()).device}")
    print(f"Имя устройства: {torch.cuda.get_device_name(0)}")

    return model

def compare_entities_and_save_diff(entities_base: dict[str, EntityInfo], entities2: dict[str, EntityInfo], result_ner_file_path: str, comment: str):
    entities2_extra_keys = list(entities2.keys() - entities_base.keys())
    entities2_extra_keys.sort(key=lambda v: v)

    with open(f"{result_ner_file_path} ({comment}).txt", "w", encoding="utf-8") as f:
        f.write("\n".join(entities2_extra_keys))



# Создаёт заданную директорию
def make_dir(dir_name):
    folder_path = Path(dir_name)
    folder_path.mkdir(parents=True, exist_ok=True)

# Сохраняет сущности в виде JSON
def save_ner(entities: dict[str, EntityInfo], file_path: str):
    entities_list = list(entities.values())
    entities_list.sort(key=lambda e: e.text)

    def convert_to_dict(e: EntityInfo) -> dict:
        ed = asdict(e)
        ed["labels"] = sorted(ed["labels"])
        return ed

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([convert_to_dict(e) for e in entities_list], f, ensure_ascii=False, indent=4)

# Загружает сущности из JSON
def load_ner(file_path: str):
    entities: dict[str, EntityInfo] = {}

    with open(file_path, "r", encoding="utf-8") as f:
        jcontent = json.load(f)

    for jitem in jcontent:
        entity = EntityInfo(text=jitem["text"], labels=set(jitem.get("labels", [])), fake_text="")
        entities[entity.text] = entity

    return entities

# Сохраняет позиции сущностей в виде JSON
def save_entity_positions(entity_positions: list[EntityDocPosInfo], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in entity_positions], f, ensure_ascii=False, indent=4)

# Загружает позиции сущностей из JSON
def load_entity_positions(file_path: str) -> list[EntityDocPosInfo]:
    with open(file_path, encoding="utf-8") as f:
        content = json.load(f)
    positions = [EntityDocPosInfo(**p) for p in content]
    return positions

# Сохраняет список как plain text
def save_list_plaintext(values: list[str], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for v in values:
            f.write(v + "\n")

# Загружает список построчно из plain text
def load_list_plaintext(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# Сохраняет документ в файл
def save_doc_to_file(file_path: str, contents: str):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(contents)
    except Exception as err:
        print(f"При сохранении документа произошла ошибка: {err}.", file=sys.stderr)



# Удаляет сущности, состоящие из стоп-слов
class EntityStopwordFilter:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.spacy_poss = {"NOUN", "ADJ", "VERB", "ADV"}
        self.digits_and_seps = frozenset("0123456789.,_-`")
        custom_stopw_path = str(Path(__file__).resolve().parent / "en_top_5000.txt")
        self.custom_stopw = set(load_list_plaintext(custom_stopw_path))

    # Удаляет стоп-слова
    def remove_stop_words(self, entities: dict[str, EntityInfo]) -> list[str]:
        rem = []
        for e in list(entities.keys()):
            if self._text_is_stopword(e):
                entities.pop(e)
                rem.append(e)
        return rem
    
    # Возвращает True, если текст является стоп словом или состоит из стоп слов
    def _text_is_stopword(self, text: str):
        strip_text = text.strip()
        if len(strip_text) == 1:
            return True

        doc = self.nlp(strip_text)

        # Простая проверка
        if not doc.text:
            return True
        
        # Слова с большой буквы
        first_title = True
        for t in doc:
            if self._is_abbreviation(t):
                return False
            if self._is_capitalized(t):
                if first_title:
                    if not self._is_stopword(t):
                        return False
                else:
                    return False
            first_title = False

        # Блокируем простые слова
        for t in doc:
            if not self._is_stopword(t):
                if t.pos_ in self.spacy_poss:
                    return True

        # Блокируем числа и стоп-слова
        for t in doc:
            if not self._is_stopword(t):
                return False
            
        return True

    def _is_abbreviation(self, t: Any):
        count = 0
        for c in t.text:
            if c.isupper():
                count += 1
            if count > 1:
                return True
        if count == 1 and len(t.text) == 1:
            return True
        return False
    
    def _is_capitalized(self, t: Any):
        return len(t.text) > 0 and t.text[0].isupper()
    
    def _is_stopword(self, t: Any):
        return t.is_stop or t.lemma_.lower() in self.custom_stopw or self._is_sep_number(t.text)

    def _is_sep_number(self, text: str):
        if not text:
            return False
        for c in text:
            if c not in self.digits_and_seps:
                return False
        return True



class EntityFakeTextGenerator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        custom_stopw_path = str(Path(__file__).resolve().parent / "en_top_5000.txt")
        self.custom_stopw = set(load_list_plaintext(custom_stopw_path))

        from _0_common.ai_fake_gen import AiFakeNameGenerator
        self.ai_fake_gen = AiFakeNameGenerator()
        self.ai_fake_gen.setup_label("character", f"Generate names for category 'person name'. Each name may consist of exactly 1 word.")
        self.ai_fake_gen.setup_label("event", f"Generate names for category 'event'. Events are battles, summits, incidents and other situations. Each name may consist of 1 to 3 words.")
        for label in ["species"]:
            self.ai_fake_gen.setup_label(label, f"Generate names for category '{label}'. Each name may consist of exactly 1 word.")
        for label in ["planet", "location", "vehicle", "weapon", "machine"]:
            self.ai_fake_gen.setup_label(label, f"Generate names for category '{label}'. Each name may consist of 1 or 2 words.")
        for label in ["organization", "droid"]:
            self.ai_fake_gen.setup_label(label, f"Generate names for category '{label}'. Each name may consist of 1 to 3 words.")

        self.fake = Faker("en_US")

        self.known_fake_texts: dict[str, dict[str, str]] = {}

    def generate_fake_text(self, entities: dict[str, EntityInfo]):
        # label, 
        labels_priority = {
            "character": 0,
            "organization": 3,
            "planet": 4,
            "location": 5,
            "vehicle": 6, 
            "weapon": 7, 
            "machine": 8, 
            "species": 2, 
            "droid": 1, 
            "event": 9, 
            "date": 10,
            "time": 11
        }

        prev_percent = 0

        for i, entity in enumerate(entities.values()):
            label = self._get_prior_label(entity, labels_priority)
            if label == "date":
                entity.fake_text = self.fake.date()
            elif label == "time":
                entity.fake_text = self.fake.time()
            elif label == "character":
                entity.fake_text = self.fake.first_name()
            else:
                entity.fake_text = self._generate_fake_text(label, entity.text)

            percent = int(i * 100 / len(entities))
            if percent > prev_percent:
                prev_percent = percent
                print(f"Обработано {percent}% сущностей")

    def _get_prior_label(self, entity: EntityInfo, labels_priority: dict[str, int]) -> str:
        return min(entity.labels, key=labels_priority.get)

    def _get_norm_text(self, text: str) -> str:
        doc = self.nlp(text.strip())
        return " ".join([t.lemma_ for t in doc])

    def _generate_fake_text(self, label: str, text: str) -> str:
        norm_text = self._get_norm_text(text)
        
        fake_text = self._get_known_fake_text(label, norm_text)
        if not fake_text:
            fake_text = self.ai_fake_gen.generate_fake_name(label)
        if not fake_text:
            fake_text = self._get_fallback_fake_text(label)

        self._set_known_fake_text(label, norm_text, fake_text)

        return fake_text

    def _get_fallback_fake_text(self, label: str):
        if label == "date":
            return self.fake.date()
        if label == "time":
            return self.fake.time()
        if label == "organization":
            return self.fake.company()
        if label == "location":
            return self.fake.city()
        else:
            return self.fake.first_name()

    def _get_known_fake_text(self, label: str, norm_text: str) -> str | None:
        label_fake_texts = self.known_fake_texts.get(label)
        if not label_fake_texts:
            return None
        known_fake_name = label_fake_texts.get(norm_text)
        if not known_fake_name:
            return None
        return known_fake_name

    def _set_known_fake_text(self, label: str, norm_text: str, fake_text: str):
        label_fake_texts = self.known_fake_texts.get(label)
        if not label_fake_texts:
            label_fake_texts = {}
            self.known_fake_texts[label] = label_fake_texts
        label_fake_texts[norm_text] = fake_text

    def _is_stopword(self, t: Any):
        return t.is_stop or t.lemma_.lower() in self.custom_stopw or self._is_sep_number(t.text)

    def save_known_fake_texts(self):
        with open("G:\\Edu\\_sprint_7\\_work_data\\ner\\known_fake_texts.txt", "w", encoding="utf-8") as f:
            for label, label_fake_texts in self.known_fake_texts.items():
                f.write(f"{label}\n")
                for norm_text, fake_text in label_fake_texts.items():
                    f.write(f"    {norm_text}={fake_text}\n")



class EntityDocumentSubstitutor:
    def substitute_entities_to_all_documents(self, docs_fold_path: str, entities: dict[str, EntityInfo]):
        for file_path in Path(docs_fold_path).rglob("*.html.md"):
            if file_path.is_file():
                print(f"{datetime.now().isoformat()} - Обработка файла '{file_path}'")
                try:
                    file_text = file_path.read_text(encoding="utf-8", errors="replace")
                    for entity in entities.values():
                        file_text = file_text.replace(entity.text, entity.fake_text)
                        save_doc_to_file(file_path.with_suffix(".anon.md"), file_text)
                except Exception as err:
                    print(f"При обработке файла произошла ошибка: {err}", file=sys.stderr)



if __name__ == "__main__":
    docs_fold_path = sys.argv[1]
    models_fold_path = sys.argv[2]
    ner_folder_path = sys.argv[3]

    os.environ["HF_HOME"] = models_fold_path

    make_dir(ner_folder_path)
    model_ner_file_path = os.path.join(ner_folder_path, "entities_v1_model.json")

    # model = init_model("gliner-community/gliner_large-v2.5", models_fold_path)
    # entities = extract_all_documents_ner(docs_fold_path, model, ENTITY_THRESHOLD)
    # save_ner(entities, model_ner_file_path)

    entities = load_ner(model_ner_file_path)
    print(f"Извлечение сущностей из текстов: {len(entities)} сущностей")

    stopw_filter = EntityStopwordFilter()
    removed_entities = stopw_filter.remove_stop_words(entities)
    stopw_ner_file_path = os.path.join(ner_folder_path, "entities_v2_after_stopw.json")
    stopw_rem_file_path = os.path.join(ner_folder_path, "entities_v2_removed.txt")
    save_ner(entities, stopw_ner_file_path)
    save_list_plaintext(removed_entities, stopw_rem_file_path)
    print(f"Удаление стоп-слов и не именных фраз: {len(entities)} сущностей")

    print(f"Генерация фейковых названий")
    fake_text_gen = EntityFakeTextGenerator()
    fake_text_gen.generate_fake_text(entities)
    fake_text_gen.save_known_fake_texts()

    print(f"Подстановка фейковых названий в документы")
    entity_substitutor = EntityDocumentSubstitutor()
    entity_substitutor.substitute_entities_to_all_documents(docs_fold_path, entities)
    
    # Эксперименты
    # model = init_model("gliner-community/gliner_small-v2.5", models_fold_path)
    # entities = extract_all_documents_ner(docs_fold_path, model, ENTITY_THRESHOLD)
    # save_ner(entities, result_ner_file_path)

    # entities_04 = extract_all_documents_ner(docs_fold_path, model, 0.4)
    # compare_entities_and_save_diff(entities, entities_04, result_ner_file_path, "small 0.4")

    # entities_03 = extract_all_documents_ner(docs_fold_path, model, 0.3)
    # compare_entities_and_save_diff(entities, entities_03, result_ner_file_path, "small 0.3")

    # model_medium = init_model("gliner-community/gliner_medium-v2.5", models_fold_path)
    # entities_medium = extract_all_documents_ner(docs_fold_path, model_medium, ENTITY_THRESHOLD)
    # compare_entities_and_save_diff(entities, entities_medium, result_ner_file_path, "medium")
   
    # model_large = init_model("gliner-community/gliner_large-v2.5", models_fold_path)
    # entities_large = extract_all_documents_ner(docs_fold_path, model_large, ENTITY_THRESHOLD)
    # compare_entities_and_save_diff(entities, entities_large, result_ner_file_path, "large")

    #model2 = init_model("gliner-community/gliner_xxl-v2.5", models_fold_path)
