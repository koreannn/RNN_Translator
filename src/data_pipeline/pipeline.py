"""원본 데이터 -> 공통 스키마 -> 중복 제거 -> 태그 -> train/valid/test 분할.

한 번 실행해 data/splits/*.jsonl을 만들고, hf_upload.py로 올린다.
실행: python -m src.data_pipeline.pipeline
"""
import argparse
import collections
import json
import random
import re
import unicodedata
from pathlib import Path

from loguru import logger

SEED = 123
TEST_DOMAIN = "해외고객과의채팅"   # test는 이 도메인에서만 뽑는다
N_TEST_NATURAL = 3000              # 실제 분포용 test
PER_CELL = 150                     # 구간별 비교용 test: (서브도메인 x 길이구간) 칸마다 뽑는 수
N_VALID = 10000
FLAGS = ("has_number", "has_negation", "has_condition")
EXCLUDED_IDS_FILE = Path(__file__).with_name("excluded_ids.txt")  # 정렬 오류로 확인된 행


# ---------------------------------------------------------------- 1. 출처별 전처리
# 출처마다 함수 하나. 반환은 공통 스키마(korean, english, source, domain, id, subdomain)의 dict 목록.

def preprocess_aihub(path):
    with open(path, encoding = "utf-8") as f:
        raw = json.load(f)["data"]
    return [
        {
            "korean": r["ko"],       # mt(기계번역 초벌)가 아니라 사람이 교정한 ko
            "english": r["en"],
            "source": "aihub",       # 원본에도 source 키가 있지만 다른 의미라 덮어씀
            "domain": r["domain"],
            "id": r["sn"],
            "subdomain": r["subdomain"],
        }
        for r in raw
    ]


# 출처를 추가할 때는 같은 스키마를 반환하는 함수를 만들어 main()의 rows에 더하면 된다.
# def preprocess_wiki(): ...


def load_excluded_ids(path = EXCLUDED_IDS_FILE):
    if not path.exists():
        return set()
    with open(path, encoding = "utf-8") as f:
        return {line.split("#")[0].strip() for line in f if line.split("#")[0].strip()}


# ---------------------------------------------------------------- 2. 공통 정제

def normalize(text):
    """중복 판정용: 대소문자·공백·문장부호를 무시한 문자열."""
    return "".join(c for c in unicodedata.normalize("NFKC", text).lower() if c.isalnum())


def deduplicate(rows):
    """한-영 쌍이 같은 행은 처음 것만 남긴다. 한쪽만 같은 행은 다른 번역 사례로 보고 유지."""
    rows = sorted(rows, key = lambda r: r["domain"])  # 어느 행을 남길지가 입력 순서에 좌우되지 않도록 순서 고정 (안정 정렬)
    seen, kept = set(), []
    for r in rows:
        key = (normalize(r["korean"]), normalize(r["english"]))
        if key not in seen:
            seen.add(key)
            kept.append(r)
    logger.info(f"중복 제거: {len(rows)} -> {len(kept)}")
    return kept


# ---------------------------------------------------------------- 3. 태그 (부정·조건은 영어에서 판별)

_NEG_EXCLUDE = re.compile(  # 부정이 아닌 no/not: 관용구, 대답으로 쓴 문두의 No, 번호(No. 1)
    r"\bno (?:problem|worries|wonder|doubt|matter)\b|\bnot only\b|\bwhy (?:don['’]t|not)\b"
    r"|^\s*no[,.!]\s*|\bno\.\s*\d", re.I)
_NEG = re.compile(r"\b(?:not|never|none|nothing|nobody|nowhere|neither|nor|cannot|without|no)\b|n['’]t\b", re.I)
_COND_EXCLUDE = re.compile(  # '~인지'를 뜻하는 if, as if
    r"\b(?:wonder(?:ing|ed)?|know(?:ing)?|see|check(?:ing)?|ask(?:ed|ing)?|request(?:ed)?|sure|confirm|find out|tell me|as)\s+if\b", re.I)
_COND = re.compile(r"\b(?:if|unless|in case|as long as|provided that)\b", re.I)
_CLAUSE = re.compile(
    r"\b(?:because|although|though|while|when|if|unless|since|whereas|but|so that|which|who|after|before|until|as soon as)\b"
    r"|,\s*(?:and|so|or)\b|;", re.I)
_DIGIT = re.compile(r"\d")


def add_tags(row):
    ko, en = row["korean"], row["english"]
    row["en_words"] = len(en.split())
    row["ko_chars"] = len(ko)
    row["has_number"] = bool(_DIGIT.search(ko) or _DIGIT.search(en))
    row["has_negation"] = bool(_NEG.search(_NEG_EXCLUDE.sub(" ", en)))
    row["has_condition"] = bool(_COND.search(_COND_EXCLUDE.sub(" ", en)))
    row["n_clauses"] = 1 + len(_CLAUSE.findall(en))  # 어림값, 분할 기준으로는 쓰지 않음
    return row


def length_bucket(en_words):
    return "short" if en_words <= 8 else ("mid" if en_words <= 16 else "long")


# ---------------------------------------------------------------- 4. 분할

def pick_balanced(rows, candidates, rng):
    """(서브도메인 x 길이구간) 칸마다 PER_CELL행. 절반은 숫자/부정/조건 문장(3등분), 절반은 특성 없는 문장."""
    cells = collections.defaultdict(list)
    for i in candidates:
        cells[(rows[i]["subdomain"], length_bucket(rows[i]["en_words"]))].append(i)

    per_flag = PER_CELL // 2 // len(FLAGS)
    picked = set()
    for key in sorted(cells):
        pool = cells[key]
        rng.shuffle(pool)
        chosen = []
        for flag in FLAGS:
            chosen += [i for i in pool if rows[i][flag] and i not in chosen][:per_flag]
        plain = [i for i in pool if not any(rows[i][f] for f in FLAGS)]
        chosen += plain[:PER_CELL - len(chosen)]
        if len(chosen) < PER_CELL:  # 특성 문장이 모자란 칸은 같은 칸의 나머지로 채움
            taken = set(chosen)
            chosen += [i for i in pool if i not in taken][:PER_CELL - len(chosen)]
        picked.update(chosen)
    return picked


def replace_excluded(rows, chat, natural, balanced, excluded):
    """test에 뽑힌 제외 대상 행을 같은 조건의 행으로 바꾼다. 나머지 test 구성은 그대로 둔다.

    natural은 같은 도메인의 아무 행, balanced는 서브도메인·길이구간·숫자/부정/조건 태그가 모두 같은 행으로 교체해
    칸별 개수와 태그 구성이 유지되게 한다. 분할용 난수와 따로 써서 다른 test 행 선택에는 영향이 없다.
    (교체로 학습 후보 목록이 바뀌므로 valid 추출 결과는 달라진다)
    """
    rng = random.Random(SEED + 1)
    test = natural | balanced
    keys = {i: (normalize(rows[i]["korean"]), normalize(rows[i]["english"])) for i in chat}
    used_ko = {keys[i][0] for i in test}
    used_en = {keys[i][1] for i in test}
    profile = lambda i: (rows[i]["subdomain"], length_bucket(rows[i]["en_words"]), *(rows[i][f] for f in FLAGS))

    def is_free(j):
        return (j not in test and rows[j]["id"] not in excluded
                and keys[j][0] not in used_ko and keys[j][1] not in used_en)

    replaced = []
    for part, name in ((natural, "test_natural"), (balanced, "test_balanced")):
        for i in sorted(i for i in part if rows[i]["id"] in excluded):
            if name == "test_natural":
                pool = [j for j in chat if is_free(j)]
            else:
                pool = [j for j in chat if is_free(j) and profile(j) == profile(i)] \
                    or [j for j in chat if is_free(j) and profile(j)[:2] == profile(i)[:2]]  # 같은 태그 조합이 없으면 같은 칸
            j = rng.choice(pool)
            part.discard(i)
            part.add(j)
            test.add(j)
            used_ko.add(keys[j][0])
            used_en.add(keys[j][1])
            replaced.append((name, rows[i]["id"], rows[j]["id"]))
    for name, old, new in replaced:
        logger.info(f"{name}: {old} -> {new}")
    return natural, balanced


def split(rows, excluded = frozenset()):
    rng = random.Random(SEED)
    chat = [i for i, r in enumerate(rows) if r["domain"] == TEST_DOMAIN]

    natural = set(rng.sample(chat, N_TEST_NATURAL))
    balanced = pick_balanced(rows, [i for i in chat if i not in natural], rng)
    if excluded:
        natural, balanced = replace_excluded(rows, chat, natural, balanced, excluded)
    test = natural | balanced

    # test와 한국어 또는 영어가 같은 문장은 학습에서 뺀다 (번역 방향과 무관하게 입력 문장 누수 방지)
    test_ko = {normalize(rows[i]["korean"]) for i in test}
    test_en = {normalize(rows[i]["english"]) for i in test}
    rest, dropped, misaligned = [], [], []
    for i, r in enumerate(rows):
        if i in test:
            continue
        if r["id"] in excluded:
            misaligned.append(i)
            continue
        overlap = normalize(r["korean"]) in test_ko or normalize(r["english"]) in test_en
        (dropped if overlap else rest).append(i)

    # valid: 도메인 비율대로, 나머지는 train
    by_domain = collections.defaultdict(list)
    for i in rest:
        by_domain[rows[i]["domain"]].append(i)
    valid = set()
    for domain in sorted(by_domain):
        valid.update(rng.sample(by_domain[domain], round(N_VALID * len(by_domain[domain]) / len(rest))))
    train = [i for i in rest if i not in valid]
    rng.shuffle(train)  # 앞에서 잘라 써도 분포가 유지되도록 섞어 둠

    take = lambda idx: [rows[i] for i in idx]
    return {
        "train": take(train),
        "valid": take(sorted(valid)),
        "test_natural": take(sorted(natural)),
        "test_balanced": take(sorted(balanced)),
        "dropped_test_overlap": take(dropped),
        "dropped_misaligned": take(sorted(misaligned)),
    }


def verify(splits, total, excluded):
    """split 간 id 중복, 행 수 합계, test 원문 누수, 제외 행 유입을 확인. 하나라도 어긋나면 중단."""
    ids = [r["id"] for rows in splits.values() for r in rows]
    assert len(ids) == len(set(ids)) == total, "split 간 id 중복 또는 행 수 불일치"
    test_rows = splits["test_natural"] + splits["test_balanced"]
    test_ko = {normalize(r["korean"]) for r in test_rows}
    test_en = {normalize(r["english"]) for r in test_rows}
    for name in ("train", "valid"):
        assert not any(normalize(r["korean"]) in test_ko or normalize(r["english"]) in test_en for r in splits[name]), f"{name}에 test 원문 누수"
    for name in ("train", "valid", "test_natural", "test_balanced"):
        assert not any(r["id"] in excluded for r in splits[name]), f"{name}에 제외 대상 행이 남아 있음"


# ---------------------------------------------------------------- 5. 실행

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aihub", default = "data/일상생활및구어체_영한_train_set.json")
    parser.add_argument("--out-dir", default = "data/splits")
    args = parser.parse_args()

    excluded = load_excluded_ids()
    logger.info(f"제외 대상 ID: {len(excluded)}개 ({EXCLUDED_IDS_FILE.name})")

    rows = preprocess_aihub(args.aihub)        # 출처가 늘면 여기서 더한다
    rows = [add_tags(r) for r in deduplicate(rows)]
    splits = split(rows, excluded)
    verify(splits, len(rows), excluded)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    for name, part in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding = "utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii = False) + "\n")
        logger.info(f"{name}: {len(part)}행")


if __name__ == "__main__":
    main()
