from piighost.pipeline import AnonymizationPipeline
import importlib.util
from piighost.components.detector.ner.base import BaseNERDetector
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
        self.rng = random.Random(42)
        self.syllables = self._create_syllables(256, self.rng)

    def _create_syllables(self, count: int, rng: random.Random):
        vowels = ['a', 'e', 'i', 'o', 'u']
        consonants = [
            'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z'
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

    def __init__(self):
        self.mapping: dict[str, str] = {}
        self.all_pseudonyms = set()
        self._fake = Faker('en-US')
        self._custom_fake = CustomFaker()
        self.stopwords = set(self._load_stopwords("en_top_5000.txt"))
        self.nlp = spacy.load("en_core_web_sm")

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
        # Проверяем многострочный текст
        if " " in entity_text:
            return " ".join([self._get_pseudonym(w, label) for w in entity_text.split()])

        # Определяем нормальную форму
        norm_entity_text = entity_text.lower()
        doc = self.nlp(norm_entity_text)
        if len(doc) > 0:
            norm_entity_text = " ".join([term.lemma_ for term in doc])

        # Проверяем исключений
        if norm_entity_text in self.stopwords:
            return entity_text
        for term in doc:
            if term.lemma_ in self.stopwords:
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
        if label == "person":
            return self._fake.first_name()

        if label == "first_name":
            return self._fake.first_name()
        if label == "middle_name":
            return self._fake.first_name()
        if label == "last_name":
            return self._fake.last_name()

        if label == "droid":
            return self._fake.first_name()

        if label == "organization":
            return self._fake.company()

        return self._custom_fake.generate()


class TextAnonimizer:
    def __init__(self, gliner2_model_name: str, gliner2_model_inp_tokens: int | None):
        # NER-детектор с произвольным набором меток
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        detector = MyGliner2Detector(
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
            threshold=0.45,
            map_location=device
        )

        # Фабрика псевдонимов
        self.preudonym_factory = MyPseudonymPlaceholderFactory()

        # Анонимайзер
        anonymizer = Anonymizer(self.preudonym_factory)

        # Пайплайн анонимизации
        self.pipeline = AnonymizationPipeline(detector=detector, anonymizer=anonymizer)

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
                result = await self.pipeline.anonymize(chunk)
                chunks[j] = result.text
            lines[i] = "".join(chunks)

        return "\n".join(lines)
