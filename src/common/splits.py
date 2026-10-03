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
