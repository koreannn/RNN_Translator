import argparse
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import mlflow
import numpy as np
import requests
from loguru import logger

from src.seq2seq.utils import load_config
from src.common.splits import load_splits, get_test_examples

UNIT_MB = {"B": 1 / 1024 ** 2, "kB": 1 / 1024, "KiB": 1 / 1024, "MB": 1, "MiB": 1, "GB": 1024, "GiB": 1024}


def container_mem_mb(container): # docker stats의 "1.9GiB / 7.6GiB"에서 앞부분을 MB로
    out = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container],
        capture_output = True, text = True,
    ).stdout.strip()
    match = re.match(r"([\d.]+)\s*([A-Za-z]+)", out.split("/")[0].strip())
    return float(match.group(1)) * UNIT_MB[match.group(2)] if match else None


class MemorySampler(threading.Thread): # 측정하는 동안 1초마다 메모리를 읽어 최대값 기록
    def __init__(self, container):
        super().__init__(daemon = True)
        self.container, self.peak_mb = container, 0.0
        self.stop_event = threading.Event() # Thread 내부 메서드 _stop과 이름이 겹치지 않게

    def run(self):
        while not self.stop_event.is_set():
            mb = container_mem_mb(self.container)
            if mb is not None:
                self.peak_mb = max(self.peak_mb, mb)
            self.stop_event.wait(1)

    def stop(self):
        self.stop_event.set()
        self.join()


def translate(api_url, text, decoding): # 요청 하나 → (번역문, latency ms)
    payload = {"dataframe_records": [{"text": text}], "params": {"decoding": decoding}}
    start = time.perf_counter()
    response = requests.post(f"{api_url}/invocations", json = payload, timeout = 300)
    latency_ms = (time.perf_counter() - start) * 1000 # 서버에서 기다린 시간(대기열) 포함
    response.raise_for_status()
    return response.json()["predictions"][0]["translation"], latency_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type = int, required = True) # 컨테이너에 준 MLFLOW_MODELS_WORKERS (기록용)
    parser.add_argument("--threads", type = int, required = True) # 컨테이너에 준 OMP_NUM_THREADS (기록용)
    parser.add_argument("--container", default = "rnn-bench")
    parser.add_argument("--concurrency", type = int, default = 8) # 동시 요청 수 (프로세스 수보다 넉넉하게)
    parser.add_argument("--num-requests", type = int, default = 50) # test 앞 N문단
    parser.add_argument("--warmup", type = int, default = 5)
    parser.add_argument("--decoding", default = "greedy")
    args = parser.parse_args()

    config = load_config("config/config.yaml")
    api_url = config["serving"]["api_url"]
    examples = get_test_examples(load_splits(config["data"], config["seed"]), args.num_requests)
    texts = [e["source"] for e in examples]

    for text in texts[: args.warmup]: # 첫 요청은 느리므로 측정에서 제외
        translate(api_url, text, args.decoding)

    sampler = MemorySampler(args.container)
    sampler.start()
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers = args.concurrency) as pool: # 동시에 concurrency개씩 요청
        results = list(pool.map(lambda t: translate(api_url, t, args.decoding), texts))
    elapsed = time.perf_counter() - start
    sampler.stop()

    latencies = np.array([ms for _, ms in results])
    metrics = {
        "throughput_sent_per_sec": len(texts) / elapsed,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_mean_ms": float(latencies.mean()),
        "peak_mem_mb": sampler.peak_mb,
        "empty_translation_count": sum(not t.strip() for t, _ in results),
    }
    logger.info(f"workers = {args.workers}, threads = {args.threads}: " + ", ".join(f"{k} = {v:,.1f}" for k, v in metrics.items()))

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run(
        run_name = f"serve-benchmark-w{args.workers}-t{args.threads}",
        tags = {"run_type": "serving_benchmark", "model_family": "rnn", "decoding": args.decoding, "arch": "arm64"},
    ):
        mlflow.log_params({
            "workers": args.workers,
            "omp_num_threads": args.threads,
            "concurrency": args.concurrency,
            "num_requests": args.num_requests,
            "warmup": args.warmup,
            "model_uri": "models:/rnn-translator@champion",
        })
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"latencies_ms": latencies.round(3).tolist()}, "latencies.json") # 요청별 원본