import mlflow
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm.prompt import build_prompt
from src.llm.postprocess import clean_translation


def _pick_device(): # GPU(cuda) > Mac GPU(mps) > CPU
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32 # CPU는 bf16 행렬 곱이 느린 경우가 많아 float32


class LoRATranslator(mlflow.pyfunc.PythonModel):
    # 한국어 → 영어 번역 (Qwen3-1.7B + LoRA adapter). 평가 때와 같은 프롬프트·후처리 사용

    def load_context(self, context):
        self.device, dtype = _pick_device()
        self.cfg = context.model_config # base_model, base_model_revision, max_new_tokens, prompt, enable_thinking
        adapter_dir = context.artifacts["adapter"]

        # build_prompt가 기대하는 형태로 맞춤 (평가 config의 llm 섹션과 같은 구조)
        self.llm_cfg = {
            "prompt": self.cfg["prompt"],
            "generation": {"enable_thinking": self.cfg["enable_thinking"]},
        }

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        base = AutoModelForCausalLM.from_pretrained(
            self.cfg["base_model"],
            revision = self.cfg["base_model_revision"],
            dtype = dtype,
        )
        model = PeftModel.from_pretrained(base, adapter_dir)
        self.model = model.merge_and_unload()
        self.model.to(self.device)
        self.model.eval() # dropout 끄기

    def _translate(self, text):
        prompt = build_prompt(text, self.tokenizer, self.llm_cfg)
        inputs = self.tokenizer(prompt, return_tensors = "pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens = self.cfg["max_new_tokens"],
            do_sample = False, # greedy (Qwen3 기본 생성은 샘플링이라서 명시 필요)
        )
        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][prompt_len:] # generated 결과에는 프롬프트도 포함됨 -> 생성된 부분만 잘라내기
        return clean_translation(self.tokenizer.decode(generated, skip_special_tokens = True))

    def predict(self, context, model_input, params = None): # 입력: "text" 열 DataFrame → 출력: "translation" 열
        return pd.DataFrame({"translation": [self._translate(text) for text in model_input["text"]]})