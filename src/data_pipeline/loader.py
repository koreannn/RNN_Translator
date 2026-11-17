"""모든 모델이 데이터를 받는 유일한 진입점.

- 학습: load_splits(config) -> {"train", "valid"}, 컬럼은 source / target (+ 메타·태그)
- 평가: get_eval_examples(config, sample_size) -> [{"id", "source", "reference", 태그...}]

토큰화·프롬프트 구성은 여기 두지 않는다. 이 모듈은 모델 코드를 import하지 않는다.
"""
from datasets import Dataset, load_dataset
from loguru import logger

DIRECTIONS = {  # 방향 -> (입력 컬럼, 정답 컬럼)
    "ko2en": ("korean", "english"),
    "en2ko": ("english", "korean"),
}
TEST_SETS = {  # evaluation.test_set 값 -> 리포의 split 이름
    "natural": "test_natural",
    "balanced": "test_balanced",
}
TAG_COLUMNS = ["domain", "subdomain", "en_words", "ko_chars", "has_number", "has_negation", "has_condition", "n_clauses"]


def _columns(config):
    direction = config["data"]["direction"]
    if direction not in DIRECTIONS:
        raise ValueError(f"알 수 없는 direction: {direction} ({' | '.join(DIRECTIONS)})")
    return DIRECTIONS[direction]  # (입력 컬럼, 정답 컬럼)


def _load(config, split):
    return load_dataset(config["data"]["dataset"], split = split)  # 필요한 split 하나만 Dataset으로 반환


def sample(split: Dataset, size: int | None, seed: int) -> Dataset:
    if size is None or size > len(split):
        return split
    else:
        return split.shuffle(seed = seed).select(range(size))


def load_splits(config): # train / valid 데이터셋 반환
    source_col, target_col = _columns(config)
    domains = config["data"].get("train_domains")  # None이면 전체 도메인

    splits = {}
    for name in ("train", "valid"):
        split = _load(config, name)
        if domains:
            split = split.filter(lambda batch: [d in domains for d in batch], input_columns = "domain", batched = True)
        split = split.rename_column("source", "origin")  # 출처("aihub") 컬럼을 먼저 비켜둔다
        splits[name] = split.rename_columns({source_col: "source", target_col: "target"})
        logger.info(f"{name}: {len(splits[name])}행")
    return splits


def get_eval_examples(config, sample_size = None):
    test_set = config["evaluation"]["test_set"]
    if test_set not in TEST_SETS:
        raise ValueError(f"알 수 없는 test_set: {test_set} ({' | '.join(TEST_SETS)})")
    source_col, reference_col = _columns(config)

    # 파일이 서브도메인별로 몰린 순서라 앞에서 자르면 치우친다 -> sample로 섞어서 추출
    test = sample(_load(config, TEST_SETS[test_set]), sample_size, config["seed"])
    return [
        {"id": row["id"], "source": row[source_col], "reference": row[reference_col], **{k: row[k] for k in TAG_COLUMNS}}
        for row in test
    ]
