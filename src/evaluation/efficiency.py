import math
import torch
import time
import numpy as np

from huggingface_hub import get_safetensors_metadata
from transformers import AutoConfig


DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def hf_model_stats(model_name, dtype): # vLLM처럼 모델 객체에 접근하기 어려울 때: 가중치 파일 헤더로 계산
    metadata = get_safetensors_metadata(model_name)
    num_params = sum(metadata.parameter_count.values())

    # 입력 임베딩과 출력층이 가중치를 공유하는데(tie_word_embeddings) 파일에는 lm_head가 복사본으로 저장된 경우
    # (예: Qwen3-0.6B·1.7B) 실제 로드 시에는 하나로 합쳐지므로 중복분을 뺌
    tensors = {name: info for file in metadata.files_metadata.values() for name, info in file.tensors.items()}
    if AutoConfig.from_pretrained(model_name).tie_word_embeddings and "lm_head.weight" in tensors:
        num_params -= math.prod(tensors["lm_head.weight"].shape)

    return {
        "num_params": num_params,
        "weight_vram_mb": num_params * DTYPE_BYTES[dtype] / 1024 ** 2, # 로드된 가중치가 차지하는 GPU 메모리
    }


def _synchronize(device): # GPU 연산은 비동기라, 끝날 때까지 기다린 뒤 시간을 재야 정확함
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def measure_ms(fn, device): # fn()을 실행하고 (결과, 걸린 시간 ms)를 반환
    _synchronize(device)
    start = time.perf_counter()
    output = fn()
    _synchronize(device)
    return output, (time.perf_counter() - start) * 1000


def summarize_latency(latencies_ms, warmup = 5): # 앞의 warmup개는 CUDA 초기화 등으로 느리므로 통계에서 제외
    values = np.array(latencies_ms[warmup:] if len(latencies_ms) > warmup else latencies_ms)
    return {
        "latency_p50_ms": float(np.percentile(values, 50)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "latency_mean_ms": float(values.mean()),
    }

def count_parameters(model) -> dict: # torch nn.Module이면 RNN·HF LLM 모두 사용 가능
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "num_params": total,
        "num_trainable_params": trainable, # LoRA 비교용 (RNN은 total과 같음)
    }


def reset_peak_vram(device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def get_peak_vram_mb(device): # 마지막 reset 이후 최대 GPU 메모리 사용량 (cuda가 아니면 None)
    if device != "cuda":
        return None
    return torch.cuda.max_memory_allocated() / 1024 ** 2