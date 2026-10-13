import time
import requests
import streamlit as st

from src.seq2seq.utils import load_config

CONFIG_PATH = "config/config.yaml"
REQUEST_TIMEOUT_SEC = 120 # 긴 문단 + beam이면 수 초 이상 걸릴 수 있음

config = load_config(CONFIG_PATH)
api_url = config["serving"]["api_url"]

st.title("문장 번역기 테스트해보기")


def is_server_alive(): # 컨테이너 안의 번역 프로세스까지 떠 있는지 확인 (nginx만 살아 있으면 502)
    try:
        return requests.get(f"{api_url}/ping", timeout = 3).status_code == 200
    except requests.RequestException:
        return False


def translate(text, decoding): # Docker 서빙 API에 번역 요청 → (번역문, 걸린 시간 ms)
    payload = {"dataframe_records": [{"text": text}], "params": {"decoding": decoding}}
    start = time.perf_counter()
    response = requests.post(f"{api_url}/invocations", json = payload, timeout = REQUEST_TIMEOUT_SEC)
    elapsed_ms = (time.perf_counter() - start) * 1000 # 요청을 보내고 응답을 받기까지 (HTTP 왕복 포함)
    response.raise_for_status()
    return response.json()["predictions"][0]["translation"], elapsed_ms


if is_server_alive():
    st.caption(f"번역 서버 연결됨: {api_url}")
else:
    st.error(
        f"번역 서버({api_url})에 연결할 수 없습니다. Docker 컨테이너가 실행 중인지 확인해주세요.\n\n"
        "`docker run --rm -p 5001:8080 -e MLFLOW_MODELS_WORKERS=1 --memory=3g rnn-translator-serving`"
    )
    st.stop()

with st.form("translate_form"):
    source_text = st.text_area(
        "한국어 문장을 입력하세요",
        height = 120,
        placeholder = "e.g., 오늘 날씨가 정말 좋네요.",
    )
    decoding = st.radio("디코딩 방식", ["greedy", "beam"], horizontal = True, help = "greedy: 빠름 / beam: 품질 우선")
    submitted = st.form_submit_button("번역하기")

if submitted:
    stripped = source_text.strip()
    if not stripped:
        st.warning("번역할 문장을 입력해주세요.")
    else:
        try:
            with st.spinner("번역 중.."):
                translation, elapsed_ms = translate(stripped, decoding)
            st.session_state["last_result"] = (translation, elapsed_ms, decoding)
        except requests.HTTPError as e: # 서버가 요청을 거절 (예: 잘못된 params)
            st.error(f"번역 서버 오류 ({e.response.status_code}): {e.response.text[:300]}")
        except requests.RequestException as e: # 연결 끊김, 시간 초과
            st.error(f"번역 서버에 요청하지 못했습니다: {e}")

if "last_result" in st.session_state:
    translation, elapsed_ms, used_decoding = st.session_state["last_result"]
    st.subheader("번역 결과")
    st.write(translation)
    st.caption(f"응답 시간: {elapsed_ms:,.0f} ms ({used_decoding})")