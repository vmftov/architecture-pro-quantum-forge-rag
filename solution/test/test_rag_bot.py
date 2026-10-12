import json
import os
from pathlib import Path
import re
from openai import AsyncOpenAI
from ragas.llms import llm_factory
import requests
import yaml
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper, embedding_factory
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import AnswerCorrectness, AnswerRelevancy, Faithfulness


EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


def ask_golden_questions():
    api_url = "http://127.0.0.1:8090/rag/questions-ext"

    with Path("data/golden_questions.yaml").open("r", encoding="utf-8") as f:
        questions = yaml.safe_load(f)

    data = []
    for golden_question in questions["golden_questions"]:
        question = golden_question.get("question")

        response = requests.post(api_url, json={"question": question})
        response.raise_for_status()
        response_json = response.json()

        # Удаляем рассуждения и источники, т.к. они негативно влияют на оценку
        actual_answer = clean_answer(response_json.get("answer"))

        data.append({
            "question": question,
            "ground_truth": golden_question.get("answer"),
            "answer": actual_answer,
            "contexts": response_json.get("contexts"),
        })

    return data


def clean_answer(answer: str):
    if not answer:
        return answer

    index = answer.lower().find("# ответ")
    if index >= 0:
        answer = answer[(index + 7):]

    answer = re.sub(r"\[[^\]]*\]", "", answer)

    return answer.strip()


def create_qwen_llm():
    os.environ["DASHSCOPE_API_KEY"] = "..."

    return ChatOpenAI(
        model="qwen3.7-plus",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://....ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )


# def create_yandex_llm():
#     YANDEX_CLOUD_FOLDER = "..."
#     YANDEX_CLOUD_API_KEY = "..."
#     YANDEX_CLOUD_MODEL = "gpt://.../yandexgpt-5.1/latest"

#     client = AsyncOpenAI(
#         api_key=YANDEX_CLOUD_API_KEY,
#         base_url="https://ai.api.cloud.yandex.net/v1",
#         project=YANDEX_CLOUD_FOLDER,
#         timeout=120.0,
#     )
#     return llm_factory(
#         YANDEX_CLOUD_MODEL,
#         provider="openai",
#         client=client,
#         temperature=0
#     )


def assert_test_result(data, answer_correctness, faithfulness, answer_relevancy):
    expected_answer = data["ground_truth"].lower()
    actual_answer = data["answer"].lower()

    # Специальные случаи, когда ответа нет или он заблокирован
    special_cases = ["у меня недостаточно знаний в этой области", "ответ был заблокирован политикой безопасности"]
    for sc in special_cases:
        if sc in expected_answer and sc not in actual_answer:
            return False
        if sc in actual_answer and sc not in expected_answer:
            return False
        if sc in expected_answer and sc in actual_answer:
            return True

    # Совпадение с эталоном
    if answer_correctness < 0.7:
        return False
    # Соответствие контексту (отсутствие галлюцинаций)
    if faithfulness < 0.9:
        return False
    # Соответствие вопросу
    if answer_relevancy < 0.8:
        return False

    #
    return True


if __name__ == "__main__":
    data = ask_golden_questions()
    dataset = Dataset.from_list(data)

    llm = create_qwen_llm()
    # llm = create_yandex_llm()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    metrics = [
        AnswerCorrectness(llm=llm, embeddings=ragas_embeddings),
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=ragas_embeddings)
    ]

    report = evaluate(dataset, metrics=metrics, llm=llm)
    answer_correctness = report["answer_correctness"]
    faithfulness = report["faithfulness"]
    answer_relevancy = report["answer_relevancy"]

    results = []
    for i, d in enumerate(data):
        results.append({
            "question": d["question"],
            "expected_answer": d["ground_truth"],
            "actual_answer": d["answer"],
            "passed": assert_test_result(d, answer_correctness[i], faithfulness[i], answer_relevancy[i]),
            "answer_correctness": answer_correctness[i],
            "faithfulness": faithfulness[i],
            "answer_relevancy": answer_relevancy[i],
        })

    with Path("test_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
