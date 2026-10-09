from dataclasses import dataclass
import re
import torch
from transformers import pipeline, BitsAndBytesConfig

@dataclass
class AiFakeNameLabelData:
    prompt: str
    next: list[str]
    all: set[str]
    failed: bool

class AiFakeNameGenerator:
    def __init__(self):
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        self.generator = pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={ "quantization_config": quantization_config },
            device_map="auto"
        )

        print(f"CUDA: {torch.cuda.is_available()}") 
        print(f"Устройство: {self.generator.model.device}")
        if torch.cuda.is_available():
            print(f"Имя устройства: {torch.cuda.get_device_name(0)}")

        self.system_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a senior expert sci-fi fake entity names generator. "
                    "You will be prompted to generate entity names of different categories. "
                    "You must generate 30 high quality unique names for the category specified by the user. "
                    "Do not generate random sequence of symbols. "
                    "RULES: "
                    "1. Each generated name must be placed on a new line. "
                    "2. Each line must contain ONLY one single word (alphabetical characters only). "
                    "3. Do not comment, do not say 'Here are the names'. Output JUST the raw names. "
                    "4. DO NOT number the list. NO bullet points, NO numbers. Output only names. "
                    "5. Names cannot contain asterisks or other non-alphabetical symbols. "
                ),
            }
        ]

        self.label_fakes_cache: dict[str, AiFakeNameLabelData] = {}

    def _generate_fake_names(self, prompt: str) -> list[str]:
        messages = []
        messages.append(self.system_prompt[0])
        messages.append({ "role": "user", "content": prompt })

        output = self.generator(
            messages,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            do_sample=True
        )
        model_response = output[0]["generated_text"][-1]["content"]

        fake_names = []
        for name in model_response.split('\n'):
            name = name.strip()
            if not name:
                continue
            name = re.sub(r'^\d+[\.\)\-]?\s*', '', name) 
            if not name:
                continue
            fake_names.append(name)

        return fake_names

    def _ensure_label_data(self, label: str) -> AiFakeNameLabelData:
        data = self.label_fakes_cache.get(label)
        if not data:
            data = AiFakeNameLabelData("", [], set(), False)
            self.label_fakes_cache[label] = data
        return data

    def setup_label(self, label: str, prompt: str):
        data = self._ensure_label_data(label)
        data.prompt = prompt
        
    def generate_fake_name(self, label: str) -> str | None:
        data = self._ensure_label_data(label)
        if not data.failed:
            if not data.prompt:
                data.prompt = f"Generate names for category '{label}'."
            for _ in range(5):
                if len(data.next) > 0:
                    break
                new_names = self._generate_fake_names(data.prompt)
                for name in new_names:
                    if name not in data.all:
                        data.all.add(name)
                        data.next.append(name)
            if len(data.next) > 0:
                return data.next.pop()
            data.failed = True
        return None
