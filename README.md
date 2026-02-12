# 프로젝트 소개

RNN(Seq2Seq)으로 번역기를 만든다면 얼만큼 잘 번역할지 궁금해서 수행해본 프로젝트입니다.

RNN 구조를 기반으로 시작해서 Seq2Seq with Attention, Transformer 기반 번역기 세 가지 모두 구현하여 비교해보고, 인사이트를 정리하였습니다.

### 아키텍쳐

```mermaid
flowchart TB

  subgraph Artifacts
    CFG[config yaml and arguments]
    DATA[kor2en dataset csv or xlsx]
    CKPT[checkpoints directory model pt files]
  end

  subgraph Entry
    MAIN[main py training entry]
    INF[inference py entry]
    APP[app py demo entry]
  end

  subgraph Tokenization_and_Data
    HF[HuggingFace AutoTokenizer load every run]
    DL[dataloader load split encode and create batches]
  end

  subgraph Model
    S2S[Seq2Seq RNN model]
    ENC[Encoder Embedding then RNN]
    DEC[Decoder Embedding RNN Linear]
  end

  subgraph Training
    TR[trainer loop teacher forcing]
    LOSS[CrossEntropy loss]
    OPT[Optimizer Adam]
  end

  subgraph Inference
    GREEDY[Greedy decoding argmax loop]
    OUT[Translated text output]
  end

  CFG --> MAIN
  CFG --> INF
  CFG --> APP

  DATA --> DL
  HF --> DL
  CFG --> HF

  MAIN --> TR
  DL --> TR
  TR --> S2S
  S2S --> ENC
  S2S --> DEC
  TR --> LOSS
  TR --> OPT
  TR --> CKPT

  INF --> HF
  INF --> CKPT
  INF --> S2S
  HF --> GREEDY
  S2S --> GREEDY
  GREEDY --> OUT

  APP --> HF
  APP --> CKPT
  APP --> S2S
  APP --> GREEDY
```


<br>

### 프로젝트 디렉터리 구조

```
root/
├── __init__.py
├── __pycache__/
├── app.py
├── config.yaml
├── dataloader.py
├── inference.py
├── kor2en.csv
├── kor2en.xlsx
├── main.py
├── model.py
├── note.py
├── README.md
├── trainer.py
└── utils.py
```

<br>

### 프로젝트 아키텍쳐



### 데이터셋 출처

[AI Hub](https://aihub.or.kr/aihubdata/data/dwld.do?currMenu=115&topMenu=100&dataSetSn=126)

<br>

---

# 인사이트 정리
