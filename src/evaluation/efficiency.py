import torch

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