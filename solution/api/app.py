import logging
import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
import requests

from yandexllmclient import YandexLlmClient
from promptmanager import PromptManager
from llmresponsepostprocessor import LlmResponsePostProcessor

from common.loggingext import init_logging
from common.textsplitter import TextSplitter
from common.textencoder import TextEncoder
from common.timecounter import SumTimeCounter
from common.qdrantsearcher import QdrantSemanticSearcher

app = FastAPI(
    title="API RAG-ассистента",
    description="Позволяет получить ответы от поисковой RAG-системы",
    version="1.0.0"
)

logger_level = os.getenv("LOGGER_LEVEL", "INFO")
encoder_model_name = os.getenv("ENCODER_MODEL_NAME", "BAAI/bge-base-en-v1.5")
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
ya_llm_uri = os.getenv("YA_LLM_URI", "https://llm.api.cloud.yandex.net/foundationModels/v1/completion")
ya_llm_folder_id = os.getenv("YA_LLM_FOLDER_ID", "")
ya_llm_api_key = os.getenv("YA_LLM_API_KEY", "")
ya_llm_model_name = os.getenv("YA_LLM_MODEL_NAME", "yandexgpt-5.1/latest")
# ya_llm_model_name = os.getenv("YA_LLM_MODEL_NAME", "yandexgpt-5-lite/latest")

init_logging(logger_level)
logger = logging.getLogger(__name__)

encoder = TextEncoder(encoder_model_name, batch_size=32, show_stats=True)
splitter = TextSplitter(encoder, show_stats=True)

qdrant_man = QdrantSemanticSearcher(qdrant_url, "docs")
# collection_exists = self.searcher.collection_exists()
qdrant_man.init_collection(encoder)

prompts = PromptManager("prompts")

response_post_proc = LlmResponsePostProcessor()

llm = YandexLlmClient(
    url=ya_llm_uri,
    folder_id=ya_llm_folder_id,
    api_key=ya_llm_api_key,
    model_name=ya_llm_model_name,
    prompts=prompts
)


class RagRequest(BaseModel):
    question: str = Field(..., min_length=3, strip_whitespace=True)


class RagResponse(BaseModel):
    answer: str


@app.post("/rag/questions", response_model=RagResponse)
async def handle_rag_request(request: RagRequest):
    response_text: str | None = None
    try:
        # Отладочные команды
        if request.question == "debug:insert_malicious_context":
            prompts.insert_malicious_context()
            response_text = "Выполнено"
        elif request.question == "debug:use_insecure_prompt":
            prompts.use_insecure_prompt()
            response_text = "Выполнено"
        elif request.question == "debug:reset_options":
            prompts.reset_options()
            response_text = "Выполнено"
        else:
            embeddings = encoder.encode_text([request.question], show_progress_bar=False)
            points = qdrant_man.search(embeddings)

            response_text = llm.ask(points, request.question)
            response_text = response_post_proc.process_response(response_text, points)
    except requests.RequestException as e:
        response_text = f"Ошибка в процессе запроса: {e}"
    except (KeyError, IndexError, ValueError) as e:
        response_text = f"Ошибка при разборе ответа: {e}"
    except Exception as e:
        response_text = f"Неизвестная ошибка при обращении к LLM: {e}"

    return RagResponse(
        answer=response_text
    )


#         print("Подходящие результаты: ")
#         print("")
#         for rank, p in enumerate(points, 1):
#             print(f"Результат {rank} " + "-" * 120)
#             print(f"  Рейтинг:   {p.score:.4f}")
#             print(f"  Категория: {p.payload.get('category')}")
#             print(f"  Документ:  {p.payload.get('doc_name')}")
#             print(f"  Язык:      {p.payload.get('language')}")
#             print(f"  Текст:")
#             print(p.payload.get('text'))
#             print("")
#             print("")
