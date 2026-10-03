import re

def clean_translation(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags = re.DOTALL)  # 혹시 남은 추론 블록 제거
    text = text.strip()
    text = re.sub(r"^(english|translation|english translation)\s*:\s*", "", text, flags = re.IGNORECASE)  # 머리말 제거
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":  # 전체를 감싼 따옴표 제거
        text = text[1:-1].strip()
    return " ".join(text.split())  # 줄바꿈·연속 공백 정리