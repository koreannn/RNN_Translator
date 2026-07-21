import torch
import streamlit as st
from transformers import AutoTokenizer

from utils import load_config
from inference import load_checkpoint, get_model_from_checkpoint, translate_sentence

CONFIG_PATH = "config.yaml"

st.title("문장 번역기 테스트해보기")


def resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@st.cache_resource(show_spinner = "모델을 불러오는 중입니다...")
def load_resources():
    config = load_config(CONFIG_PATH)
    device = resolve_device()

    kor_tokenizer = AutoTokenizer.from_pretrained(config["model"]["kor_tokenizer"])
    en_tokenizer = AutoTokenizer.from_pretrained(config["model"]["en_tokenizer"])

    checkpoint = load_checkpoint(config["inference"]["checkpoint_path"], device = device)
    model = get_model_from_checkpoint(checkpoint, device = device)

    return model, kor_tokenizer, en_tokenizer, device, config


try:
    model, kor_tokenizer, en_tokenizer, device, config = load_resources()
except FileNotFoundError:
    st.error(
        "체크포인트 파일을 찾을 수 없습니다. "
        "config.yaml의 inference.checkpoint_path 경로를 확인해주세요."
    )
    st.stop()

max_length = config["inference"]["max_length"]
max_new_tokens = config["inference"]["max_new_token"]

st.caption(f"사용 중인 디바이스: {device}")

with st.form("translate_form"):
    source_text = st.text_area(
        "한국어 문장을 입력하세요",
        height = 120,
        placeholder = "e.g., 오늘 날씨가 정말 좋네요.",
    )
    submitted = st.form_submit_button("번역하기")

if submitted:
    stripped = source_text.strip()
    if not stripped:
        st.warning("번역할 문장을 입력해주세요.")
    else:
        with st.spinner("번역 중.."):
            translation = translate_sentence(
                model,
                kor_tokenizer,
                en_tokenizer,
                device,
                stripped,
                max_length,
                max_new_tokens,
            )
        st.session_state["last_translation"] = translation

if "last_translation" in st.session_state:
    st.subheader("번역 결과")
    st.write(st.session_state["last_translation"])