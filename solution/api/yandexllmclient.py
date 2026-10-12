import requests
from qdrant_client import models

from promptmanager import PromptManager


class YandexLlmClient:
    def __init__(self, url: str, folder_id: str, api_key: str, model_name: str, prompts: PromptManager):
        self.url = url
        self.folder_id = folder_id
        self.api_key = api_key
        self.model_name = model_name
        self.prompts = prompts

    def ask(self, points: list[models.ScoredPoint], user_request: str) -> str:
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
            "x-folder-id": self.folder_id
        }

        messages: list[dict[str, str]] = []

        messages.append({"role": "system", "text": self.prompts.get_system_prompt()})

        for p in self.prompts.get_few_shot_prompts():
            messages.append({"role": "user", "text": p[0]})
            messages.append({"role": "assistant", "text": p[1]})

        messages.append({"role": "user", "text": self.prompts.get_user_prompt(points, user_request)})

        payload = {
            "modelUri": f"gpt://{self.folder_id}/{self.model_name}",
            "completionOptions": {
                "stream": False,
                "temperature": 0.1,
                "maxTokens": "1000",
            },
            "messages": messages
        }

        response = requests.post(self.url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        answer = result['result']['alternatives'][0]['message']['text']

        return answer
