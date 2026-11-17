from datasets import load_dataset

files = {name: f"data/splits/{name}.jsonl"
        for name in ["train", "valid", "test_natural", "test_balanced", "dropped_test_overlap", "dropped_misaligned"]}
ds = load_dataset("json", data_files=files)
ds.push_to_hub("Hongsungjae/cs-translation-router", private=True)