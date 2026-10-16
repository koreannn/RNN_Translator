from datasets import load_dataset


def load_splits(data_config, seed): # RNN·LLM 공통: 같은 config·seed면 항상 같은 train/valid/test
    dataset = load_dataset(data_config["dataset2"])["train"]

    # 테스트셋 분리 → 나머지에서 검증셋 분리 (기존 CustomDataLoader와 동일한 순서·인자를 유지해야 같은 split이 나옴)
    split = dataset.train_test_split(test_size = data_config["dataset2_test_ratio"], seed = seed)
    train_valid, test = split["train"], split["test"]

    split = train_valid.train_test_split(test_size = data_config["dataset2_valid_ratio"], seed = seed)
    return {"train": split["train"], "valid": split["test"], "test": test}


def get_test_examples(splits, sample_size = None): # LLM 추론용: 토큰화 없이 원본 문장만
    test = splits["test"]
    n = len(test) if sample_size is None else min(sample_size, len(test))
    return [
        # id = test split 내 순번 → RNN records의 id(shuffle = False로 순서대로 부여)와 같은 문장을 가리킴
        {"id": i, "source": test[i]["korean"], "reference": test[i]["english"]}
        for i in range(n)
    ]

def get_eval_examples(config, sample_size = None): # config의 evaluation.test_set에 따라 평가 데이터 선택 (모든 모델 공통)
    test_set = (config.get("evaluation") or {}).get("test_set", "in_domain")
    
    if test_set == "in_domain":
        return get_test_examples(load_splits(config["data"], config["seed"]), sample_size)
    
    if test_set == "ood":
        # opus-100 en-ko 공식 test: 구어체·짧은 문장, 어떤 학습에도 쓰지 않은 데이터
        test = load_dataset(config["data"]["dataset1"], config["data"]["dataset1_config"])["test"]
        pairs = [(r["translation"]["ko"].strip(), r["translation"]["en"].strip()) for r in test]
        pairs = [(ko, en) for ko, en in pairs if ko and en] # 빈 문장 쌍 제외
        n = len(pairs) if sample_size is None else min(sample_size, len(pairs))
        return [{"id": i, "source": ko, "reference": en} for i, (ko, en) in enumerate(pairs[:n])]
    raise ValueError(f"알 수 없는 test_set: {test_set} (in_domain | ood)")