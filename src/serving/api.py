import time

import httpx
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src.seq2seq.utils import load_config

config = load_config("config/config.yaml")
gw_cfg = config["serving"]["gateway"]
SUPPORTED_PAIRS = {("ko", "en")} # 학습한 방향만 허용

app = FastAPI(title = "Translation API")


class TranslateRequest(BaseModel):
    text: str = Field(min_length = 1, max_length = gw_cfg["max_chars"])
    source_language: str = "ko"
    target_language: str = "en"
    model: str | None = None # 비우면 기본 모델


class TranslateResponse(BaseModel):
    translated_text: str
    model: str
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "models": list(gw_cfg["backends"])}


@app.post("/v1/translate", response_model = TranslateResponse)
def translate(req: TranslateRequest):
    # 1) 입력 처리
    if (req.source_language, req.target_language) not in SUPPORTED_PAIRS:
        raise HTTPException(400, f"지원하지 않는 언어 쌍: {req.source_language}→{req.target_language}")
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "빈 문장은 번역할 수 없습니다")
    model = req.model or gw_cfg["default_model"]
    if model not in gw_cfg["backends"]:
        raise HTTPException(400, f"없는 모델: {model}")

    # 2) 모델 추론 (MLflow 모델 서버 형식으로 바꿔서 호출)
    start = time.perf_counter()
    try:
        resp = httpx.post(
            f"{gw_cfg['backends'][model]}/invocations",
            json = {"dataframe_records": [{"text": text}]},
            timeout = gw_cfg["timeout_sec"],
        )
        resp.raise_for_status()
    except httpx.TimeoutException:
        raise HTTPException(504, f"{model} 응답 시간 초과")
    except httpx.HTTPError as e:
        logger.error(f"{model} 호출 실패: {e}")
        raise HTTPException(502, f"{model} 모델 서버 오류")
    latency_ms = (time.perf_counter() - start) * 1000

    # 3) 결과 응답 (MLflow 형식 → API 형식)
    translated = resp.json()["predictions"][0]["translation"]
    logger.info(f"model={model} chars={len(text)} latency_ms={latency_ms:.0f}")
    return TranslateResponse(translated_text = translated, model = model, latency_ms = round(latency_ms, 1))