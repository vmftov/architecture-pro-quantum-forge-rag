from pathlib import Path
from typing import Any

from piighost.pipeline import AnonymizationPipeline
import importlib.util
from piighost.components.detector.ner.base import BaseNERDetector
from piighost.components.detector import ExactMatchDetector
from piighost.components.anonymizer import Anonymizer
from piighost.components.placeholder import AnyPlaceholderFactory, PreservesLabeledIdentity
from piighost.models import Detection, Entity, Span
from faker import Faker
import torch
import random
import json
import spacy

from chunking import split_to_chunks

if importlib.util.find_spec("gliner2") is None:
    raise ImportError("Gliner2Detector requires the gliner2 package. Install it with: pip install piighost[gliner2]")

from gliner2 import GLiNER2


class MyGliner2Detector(BaseNERDetector):
    """
    Детектор сущностей. Единственное отличие от стандартного метода - параметр map_location, для запуска модели на GPU.
    """

    def __init__(
        self,
        model: GLiNER2 | str,
        labels: list[str] | dict[str, str],
        threshold: float = 0.5,
        max_concurrency: int | None = None,
        max_chars: int | None = None,
        auto_chunk: bool = True,
        map_location: str = "auto"
    ) -> None:
        """Store or load the model, then set the labels and threshold."""
        super().__init__(
            labels,
            max_concurrency=max_concurrency,
            max_chars=max_chars,
            auto_chunk=auto_chunk,
        )
        self.model = GLiNER2.from_pretrained(model, map_location=map_location) if isinstance(model, str) else model
        self.threshold = threshold

    async def _raw_detect(self, text: str) -> list[Detection]:
        """Run GLiNER2 and build one detection per entity, native labels kept."""
        result = await self._run_blocking(
            self.model.extract_entities,
            text,
            self.internal_labels,
            threshold=self.threshold,
            include_spans=True,
            include_confidence=True,
        )
        detections: list[Detection] = []
        for native_label, entities in result["entities"].items():
            for entity in entities:
                span = Span(entity["start"], entity["end"])
                detection = Detection(
                    span=span,
                    text=entity["text"],
                    label=native_label,
                    confidence=entity["confidence"],
                )
                detections.append(detection)
        return detections


class CustomFaker:
    def __init__(self):
        self.rng = random.Random(11)
        self.syllables = self._create_syllables(256, self.rng)

    def _create_syllables(self, count: int, rng: random.Random):
        vowels = ['e', 'i', 'a', 'o', 'u']
        consonants = [
            'r', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'b', 'c', 'm', 'n', 'p', 'q', 't', 'v', 'w', 'x', 'y', 'z'
        ]

        def rand_vowel():
            return rng.choice(vowels)

        def rand_consonant():
            return rng.choice(consonants)

        syllables = set()
        strats = [
            lambda: f"{rand_vowel()}",
            lambda: f"{rand_vowel()}{rand_consonant()}",
            lambda: f"{rand_vowel()}{rand_consonant()}{rand_vowel()}",
            lambda: f"{rand_consonant()}{rand_vowel()}",
            lambda: f"{rand_consonant()}{rand_vowel()}{rand_consonant()}",
            lambda: f"{rand_consonant()}{rand_vowel()}{rand_vowel()}{rand_consonant()}",
        ]

        while len(syllables) < count:
            syllables.add(strats[rng.randrange(0, len(strats))]())

        return list(syllables)

    def generate(self):
        word = ""
        for _ in range(self.rng.randrange(2, 5)):
            word += self.rng.choice(self.syllables)
        return word[0].upper() + word[1:]


class MyPseudonymPlaceholderFactory(AnyPlaceholderFactory[PreservesLabeledIdentity]):
    """
    Фабрика, которая заменяет сущности на читаемые псевдонимы и сохраняет сделанные изменения.
    """

    def __init__(self, exact_entities: dict[str, str]):
        self.mapping: dict[str, str] = {}
        self.all_pseudonyms = set()
        self._fake = Faker('en-US')
        self._custom_fake = CustomFaker()
        self.stopwords = set(self._load_stopwords("en_top_5000.txt"))
        self.exact_mult_word_entities = [e for e in exact_entities.keys() if " " in e]
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp_target_poss = {"PROPN", "NOUN", "ADJ", "VERB", "ADV"}

    def _load_stopwords(self, file_path: str) -> list[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def to_string(self) -> str:
        data = [{"orig": orig, "pseud": pseud} for orig, pseud in self.mapping.items()]
        return json.dumps(data, ensure_ascii=False, indent=2)

    def from_string(self, str: str):
        self.mapping = {}
        self.all_pseudonyms = set()

        if str:
            data = json.loads(str)
            for item in data:
                pseudonym = item["pseud"]
                self.mapping[item["orig"]] = pseudonym
                self.all_pseudonyms.add(pseudonym)

    def create(self, entities: list[Entity]) -> dict[Entity, str]:
        result: dict[Entity, str] = {}
        for entity in entities:
            result[entity] = self._get_pseudonym(entity.text, entity.label)
        return result

    def _get_pseudonym(self, entity_text: str, label: str) -> str:
        # Проверка явно заданных сущностей из нескольких слов
        if entity_text in self.exact_mult_word_entities:
            return self._get_single_pseudonym(entity_text, label)

        # Разбиение на термы
        doc = self.nlp(entity_text)
        if len(doc) == 0:
            return entity_text

        # Замена термов по одному
        result_text = ""
        for term in doc:
            if len(term.text) > 1 and term.pos_ in self.nlp_target_poss:
                result_text += self._get_single_pseudonym(term.lemma_, label)
            else:
                result_text += term.text
            result_text += term.whitespace_
        return result_text

    def _get_single_pseudonym(self, entity_text: str, label: str) -> str:
        #
        norm_entity_text = entity_text.lower()

        # Проверяем исключений
        if norm_entity_text in self.stopwords:
            return entity_text

        # Test
        # return f"<<{label}>>"

        # Проверяем наличие имеющегося псевдонима
        pseudonym = self.mapping.get(norm_entity_text)
        if pseudonym:
            return pseudonym

        # Генерируем новый псевдоним
        while True:
            pseudonym = self._generate_new_pseudonym(label)
            if pseudonym not in self.all_pseudonyms:
                break
        self.mapping[norm_entity_text] = pseudonym
        self.all_pseudonyms.add(pseudonym)

        return pseudonym

    def _generate_new_pseudonym(self, label: str):
        # if label == "person":
        #     return self._fake.first_name()

        # if label == "first_name":
        #     return self._fake.first_name()
        # if label == "middle_name":
        #     return self._fake.first_name()
        # if label == "last_name":
        #     return self._fake.last_name()

        # if label == "droid":
        #     return self._fake.first_name()

        if label == "organization":
            return self._fake.company()

        return self._custom_fake.generate()


class TextAnonimizer:
    def __init__(self, gliner2_model_name: str, gliner2_model_inp_tokens: int | None):
        # NER-детектор с произвольным набором меток
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        gliner2_detector = MyGliner2Detector(
            model=gliner2_model_name,
            labels=[
                "first_name", "middle_name", "last_name",
                # "person",
                "droid",
                "race", "ethnicity",
                "organization",
                "weapon",
                "planet", "location",
                "transport",
                "event"
            ],
            threshold=0.42,
            map_location=device
        )

        # Детектор явно заданных сущностей
        self.exact_entities = self._load_exact_entities()
        self.exact_detector = self._create_exact_detector()

        # Фабрика псевдонимов
        self.preudonym_factory = MyPseudonymPlaceholderFactory(self.exact_entities)

        # Анонимайзер
        anonymizer = Anonymizer(self.preudonym_factory)

        # Пайплайн анонимизации
        self.pipelines = [AnonymizationPipeline(detector=gliner2_detector, anonymizer=anonymizer)]
        if self.exact_detector:
            self.pipelines.append(AnonymizationPipeline(detector=self.exact_detector, anonymizer=anonymizer))

        # Определяем максимальную длину чанка
        if not gliner2_model_inp_tokens or gliner2_model_inp_tokens <= 0:
            gliner2_model_inp_tokens = 512
        self.max_chunk_size_symbols = int(gliner2_model_inp_tokens * 3 * 0.85)

    def get_terms_map(self) -> str:
        return self.preudonym_factory.to_string()

    def set_terms_map(self, str: str):
        self.preudonym_factory.from_string(str)

    async def anonymize(self, text: str):
        lines = text.split("\n")

        for i, line in enumerate(lines):
            chunks, _ = split_to_chunks(line, self.max_chunk_size_symbols)
            for j, chunk in enumerate(chunks):
                result_text = chunk
                for ppl in self.pipelines:
                    result = await ppl.anonymize(result_text)
                    result_text = result.text
                chunks[j] = result_text
            lines[i] = "".join(chunks)

        return "\n".join(lines)

    def _load_exact_entities(self) -> dict[str, str]:
        entities: dict[str, str] = {}
        exact_entities_path = Path("explicit_terms.txt")
        if exact_entities_path.exists():
            for line in exact_entities_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split("=", 1)
                if len(parts) == 2:
                    entities[parts[0]] = parts[1]
        return entities

    def _get_exact_entities_from_terms_map(self, terms_map: str):
        entities: dict[str, str] = {}
        if terms_map:
            data = json.loads(terms_map)
            for item in data:
                entities[item["orig"]] = "other"
        return entities

    def _create_exact_detector(self) -> ExactMatchDetector | None:
        if len(self.exact_entities) == 0:
            return None
        return ExactMatchDetector(self.exact_entities, case_sensitive=False)
