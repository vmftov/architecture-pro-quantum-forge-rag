from dataclasses import dataclass
import logging
import os
import statistics
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
import requests
from qdrant_client import models

from yandexllmclient import YandexLlmClient
from promptmanager import PromptManager
from naivesecurityvalidator import NaiveSecurityValidator
from llmresponsepostprocessor import LlmResponsePostProcessor

from common.loggingext import init_logging
from common.textsplitter import TextSplitter
from common.textencoder import TextEncoder
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
security_validator = NaiveSecurityValidator()
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

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        max_tokens = encoder.get_max_seq_length_tokens() * 0.85
        actual_tokens = encoder.count_tokens(value)
        if actual_tokens > max_tokens:
            raise ValueError(f"Количество символов в вопросе превышает допустимый порог. Сократите вопрос.")
        return value


class RagResponse(BaseModel):
    answer: str


class RagResponseExt(BaseModel):
    answer: str
    contexts: list[str]


@dataclass
class RagResponseImpl:
    answer: str
    contexts: list[str]


class SecurityError(Exception):
    pass


def calc_context_score_median(points: list[models.ScoredPoint]) -> float:
    if len(points) == 0:
        return 0.0
    return statistics.median([p.score for p in points])


def check_response_successful(response_text: str, source_refs: list[str]) -> bool:
    norm_response_text = response_text.lower()

    # Ответ не найден
    if "у меня недостаточно знаний в этой области" in norm_response_text:
        return False

    # В запросе или контексте есть проблемы безопасности
    if "ответ был заблокирован политикой безопасности" in norm_response_text:
        return False

    # Ответ не соответствует формату из системного промпта
    thoughs_index = norm_response_text.find("# рассуждение")
    answer_header = "# ответ"
    answer_index = norm_response_text.find(answer_header)
    if thoughs_index < 0 or answer_index < 0:
        return False

    # Ответ слишклм короткий
    answer = response_text[(answer_index + len(answer_header)):]
    if len(answer) < 10:
        return False

    # Нет подтверждающих ссылок на документы
    if len(source_refs) == 0:
        return False

    return True


async def handle_rag_request_impl(request: RagRequest) -> RagResponseImpl:
    response_text: str | None = None
    response_contexts: list[str] = []

    try:
        if not security_validator.check_request(request.question):
            logger.info(f"Небезопасный запрос '{request.question}' был отклонён")
            raise SecurityError("Ваш запрос не может быть обработан. Измените тему и содержание вопроса.")

        embeddings = encoder.encode_text([request.question], show_progress_bar=False)

        orig_points = qdrant_man.search(embeddings)
        safe_points = security_validator.filter_context(orig_points)

        if len(safe_points) > 0:
            llm_response_text = llm.ask(safe_points, request.question)

            if not security_validator.check_llm_response(llm_response_text):
                logger.info(f"Небезопасный ответ LLM '{request.question}' был отклонён")
                raise SecurityError("Ваш запрос не может быть обработан. Измените тему и содержание вопроса.")

            response_text = response_post_proc.process_response(llm_response_text, safe_points)
        else:
            response_text = "У меня недостаточно знаний в этой области"

        response_contexts = [p.payload.get('text') for p in safe_points]
        found_doc_names = [p.payload.get('doc_name') for p in orig_points]
        source_refs = response_post_proc.get_source_refs(response_text)

        logger.info(f"Вызов API", extra={
            "question": request.question,
            "orig_chunk_count": len(orig_points),
            "safe_chunk_count": len(safe_points),
            "context_score_median": calc_context_score_median(safe_points),
            "response_length": len(response_text),
            "is_success": check_response_successful(response_text, source_refs),
            "doc_names": ", ".join(found_doc_names)
        })
    except requests.RequestException as e:
        response_text = f"Ошибка в процессе запроса '{request.question}': {e}"
        logger.error(response_text)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ошибка в процессе запроса")
    except (SecurityError, ValueError) as e:
        response_text = f"{e}"
        logger.error(response_text)
    except Exception as e:
        response_text = f"Неизвестная ошибка при обработке запроса '{request.question}': {e}"
        logger.error(response_text)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Неизвестная ошибка при обработке запроса")

    return RagResponseImpl(answer=response_text, contexts=response_contexts)


@app.post("/rag/questions", response_model=RagResponse)
async def handle_rag_request(request: RagRequest):
    response_impl = await handle_rag_request_impl(request)
    return RagResponse(answer=response_impl.answer)


@app.post("/rag/questions-ext", response_model=RagResponseExt)
async def handle_rag_request_ext(request: RagRequest):
    response_impl = await handle_rag_request_impl(request)
    return RagResponseExt(answer=response_impl.answer, contexts=response_impl.contexts)
