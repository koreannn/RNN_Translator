import mlflow
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.seq2seq.model import build_model
from src.seq2seq.decoding import get_special_token_ids, greedy_decoding, beam_decoding


class RNNTranslator(mlflow.pyfunc.PythonModel):
    # 한국어 → 영어 번역. 가중치·토크나이저·디코딩 설정을 하나로 묶어, 불러온 쪽은 predict만 호출하면 됨

    def load_context(self, context): # 모델을 불러올 때 한 번 실행
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = context.model_config # log_model(model_config = ...)로 저장한 디코딩 설정

        checkpoint = torch.load(context.artifacts["checkpoint"], map_location = "cpu")
        self.model = build_model(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["seq2seq_state_dict"])
        self.model.to(self.device).eval()

        # 모델과 함께 저장된 토크나이저 사용 (서빙 시 Hugging Face Hub 접속 불필요)
        self.kor_tokenizer = AutoTokenizer.from_pretrained(context.artifacts["kor_tokenizer"])
        self.en_tokenizer = AutoTokenizer.from_pretrained(context.artifacts["en_tokenizer"])
        self.ids = get_special_token_ids(self.kor_tokenizer, self.en_tokenizer)

    def _translate(self, text, decoding): # 한 문장씩 (Encoder가 패딩을 처리하지 않아, 묶으면 결과가 달라질 수 있음)
        src_ids = self.kor_tokenizer(
            text, truncation = True, max_length = self.cfg["max_length"], return_tensors = "pt",
        )["input_ids"].to(self.device)

        if decoding == "greedy":
            gen_ids = greedy_decoding(self.model, src_ids, max_new_tokens = self.cfg["max_new_tokens"], **self.ids)[0]
        else:
            gen_ids = beam_decoding(
                self.model, src_ids,
                max_new_tokens = self.cfg["max_new_tokens"],
                max_length = self.cfg["max_length"],
                beam_size = self.cfg["beam_size"],
                alpha = self.cfg["beam_alpha"],
                **self.ids,
            )
        return self.en_tokenizer.decode(gen_ids, skip_special_tokens = True).strip()

    def predict(self, context, model_input, params = None): # 입력: "text" 열이 있는 DataFrame → 출력: "translation" 열
        decoding = (params or {}).get("decoding", self.cfg["default_decoding"])
        if decoding not in ("greedy", "beam"):
            raise ValueError(f"알 수 없는 디코딩 전략: {decoding} (greedy | beam)")
        return pd.DataFrame({"translation": [self._translate(text, decoding) for text in model_input["text"]]})