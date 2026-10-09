from pathlib import Path
from qdrant_client import models
import yaml


class PromptManager:
    def __init__(self, prompt_dir: str):
        self.prompt_dir = Path(prompt_dir)

    def get_system_prompt(self) -> str:
        return (self.prompt_dir / "system_prompt.md").read_text(encoding="utf-8", errors="replace")

    def get_few_shot_prompts(self) -> list[tuple[str, str]]:
        yaml_text = (self.prompt_dir / "few_shot_prompts.yaml").read_text(encoding="utf-8")
        yaml_list = yaml.safe_load(yaml_text)
        return [(item["user"], item["assistant"]) for item in yaml_list]

    def get_user_prompt(self, points: list[models.ScoredPoint], user_request: str) -> str:
        user_prompt = """# Контекст
"""
        for rank, p in enumerate(points, 1):
            user_prompt += f"""
## Факт {rank}

```
{p.payload.get('text')}
```

"""
        user_prompt += f"""# Вопрос пользователя

{user_request}
"""
        return user_prompt
