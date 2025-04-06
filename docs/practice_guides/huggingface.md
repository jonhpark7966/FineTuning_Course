# Hugging Face 소개 및 활용 가이드

Hugging Face는 머신러닝 모델, 데이터셋, 애플리케이션을 공유하고 배포하기 위한 플랫폼으로, LLM 이후 표준으로 자리 잡았습니다. 이 문서에서는 Hugging Face 생태계의 주요 구성 요소와 이를 실무에서 활용하는 방법을 다룹니다.

| 💡 필자의 의견 |
|---------|
| 허깅페이스가 사실상 AI 쪽에서는 github과 같은 위상이죠. 중국 쪽에는 ModelScope 이라는 대체제(?) 가 있긴 합니다, 딥러닝 시대를 보면, tfhub 이나 kaggle 도 있었지만, Transformer 를 기반으로한 모델들이 천하통일을 하고 그 모델들에 많이 사용되는 허깅페이스 라이브러리 덕분인지, 허깅페이스가 사실상 표준이 되었습니다. |


## Hugging Face 플랫폼 개요

### 모델 및 데이터셋 저장소

Hugging Face Models Hub은 1,000,000개 이상의 모델과 200,000개 이상의 데이터셋을 호스팅합니다. 각 모델이나 데이터셋은 플랫폼 내 Git 저장소에 있어 버전 관리, 커밋 기록, 이슈/토론 등의 기능을 제공합니다.

- 모델 업로드: 웹 UI나 Git, huggingface_hub Python 라이브러리를 통해 모델을 추가할 수 있습니다. 예를 들어, Transformers로 모델을 파인튜닝한 후 model.push_to_hub("username/my-model")을 실행하면 가중치가 저장소에 커밋됩니다. 모델은 공개 또는 비공개로 설정 가능합니다.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 모델과 토크나이저 로드 (또는 자체 학습한 모델 사용)
model = AutoModelForSequenceClassification.from_pretrained("llama...")
tokenizer = AutoTokenizer.from_pretrained("llama...")

# Hub에 모델 업로드 (로그인 필요)
model.push_to_hub("username/my_llama...")
tokenizer.push_to_hub("username/my_llama")
```

- 데이터셋 업로드 : 데이터셋도 유사하게 데이터 저장소로 저장됩니다. 새 데이터셋 저장소를 만들고 파일(CSV, JSONL, 이미지 등)을 업로드할 수 있습니다. 웹 인터페이스에서는 드래그 앤 드롭 업로드를 지원하며, 각 데이터셋 저장소에는 🤗 Datasets 라이브러리에서 직접 사용할 수 있는 데이터셋 스크립트나 메타데이터를 포함할 수 있습니다. 

```python
# 데이터셋을 Hugging Face Hub에 업로드하는 예시
from datasets import Dataset
import pandas as pd

# 로컬 CSV 파일에서 데이터셋 생성
df = pd.read_csv("my_data.csv")
dataset = Dataset.from_pandas(df)

# Hub에 데이터셋 업로드
dataset.push_to_hub("username/my-dataset")
```

- 모델 / 데이터셋 사용 : 위의 업로드와 마찬가지로 다운받아 사용하는 것도 당연히 가능합니다. 다음 실습 문서에서 이어서 직접 해보겠습니다. 



## Hugging Face 라이브러리 및 도구

### 핵심 라이브러리

- **🤗 Transformers**: 모델 아키텍처(수천 개의 사전 훈련된 모델 포함), 토크나이저 및 파이프라인을 위한 핵심 라이브러리
- **🤗 Datasets**: 데이터셋을 쉽게 로드하고 전처리하기 위한 라이브러리(Hub 통합 및 효율적인 디스크 관리 포함)
- **🤗 Hugging Face Hub (Python)**: 모델 및 데이터를 다운로드/업로드하고 Hub와 상호 작용하기 위한 유틸리티
- **🤗 Trainer API**: Transformers 내 포함된, 로깅/평가 등을 지원하는 고수준 트레이너 클래스
- **🤗 Accelerate**: 코드 변경 없이 분산 훈련 및 혼합 정밀도를 지원하는 라이브러리

### Transformers 라이브러리 상세

Transformers 라이브러리는 Hugging Face의 핵심 라이브러리로, 다양한 모델 아키텍처(BERT, GPT, LLaMA, Mistral 등)를 구현하고, 사전 훈련된 가중치를 로드하여 추론이나 미세 조정에 사용할 수 있게 해줍니다. NLP 작업뿐만 아니라 이미지, 오디오 작업에도 널리 사용됩니다.

#### 1. AutoModel과 AutoTokenizer

이 "auto" 클래스들은 모델 이름이나 경로를 기반으로 어떤 아키텍처든 로드할 수 있습니다. 예를 들어, `AutoModel.from_pretrained("mistralai/Mistral-7B-v0.1")`를 호출하면:

- Hub에서 모델 구성을 다운로드합니다(모델 유형 = Mistral).
- 해당 구성으로 내부적으로 MistralModel 클래스를 인스턴스화합니다.
- 가중치를 다운로드하고 모델 인스턴스에 로드합니다.

다양한 작업에 맞는 AutoModel 변형이 있습니다:
- `AutoModel`: 기본 모델만 제공
- `AutoModelForCausalLM`: GPT나 LLaMA 같은 언어 모델용

마찬가지로, `AutoTokenizer.from_pretrained(name)`는 모델 구성에 따라 적절한 토크나이저(LlamaTokenizer 등)를 로드합니다.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("qwen/qwen2.5-3b-instruct")
model = AutoModelForSequenceClassification.from_pretrained("qwen/qwen2.5-3b-instruct")

# 텍스트 처리 및 추론
text = "Hugging Face는 정말 놀라운 도구를 제공합니다!"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
logits = outputs.logits
print(logits)  # 클래스에 대한 원시 점수
```

#### 2. 토크나이저

`AutoTokenizer`(또는 `LlamaTokenizer`와 같은 특정 토크나이저)는 전처리를 담당합니다:
- 텍스트를 토큰 ID로 변환
- 특수 토큰 추가([CLS], [SEP], [BOS], [EOS] 등)
- 패딩/자르기 처리
- 텍스트로 다시 변환(디코딩)

각 모델 유형마다 다른 토크나이징 방식을 사용합니다:
- GPT-2: 바이트 페어 인코딩(BPE)
- LLaMA/Mistral: SentencePiece 모델
- BERT: WordPiece

토크나이저는 Hub의 모델과 함께 저장되므로, `from_pretrained`를 사용하면 모델 훈련에 사용된 것과 동일한 어휘와 전처리를 사용할 수 있습니다.

#### 3. 파이프라인

Transformers는 일반적인 작업을 위한 고수준 파이프라인 API도 제공합니다. 예를 들어, 감성 분석 파이프라인은 원시 텍스트를 받아 점수와 함께 "긍정" 또는 "부정"을 출력합니다. 파이프라인은 내부적으로 토큰화, 모델 순전파, 디코딩을 한 번의 호출로 처리합니다.

```python
from transformers import pipeline

# 감성 분석 파이프라인 생성
classifier = pipeline("sentiment-analysis")
result = classifier("Hugging Face 라이브러리를 사용하는 것이 정말 좋아요!")[0]
print(result)
# {'label': 'POSITIVE', 'score': 0.9998}

# 텍스트 생성 파이프라인
generator = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf")
result = generator("인공지능의 미래는", max_length=50, do_sample=True)
print(result[0]['generated_text'])
```



#### 4. 텍스트 생성 예제

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", 
                                            torch_dtype=torch.float16,  # 메모리 절약을 위한 반정밀도
                                            device_map="auto")  # 자동 장치 매핑
model.eval()  # 추론 모드로 설정

# 프롬프트 준비
prompt = "<s>[INST] 인공지능의 미래에 대해 간략히 설명해주세요. [/INST]"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

# 텍스트 생성
outputs = model.generate(
    **inputs, 
    max_new_tokens=200,  # 생성할 최대 토큰 수
    do_sample=True,      # 샘플링 사용
    temperature=0.7,     # 온도 설정 (낮을수록 더 결정적)
    top_p=0.9            # 누적 확률이 0.9가 될 때까지의 토큰만 고려
)

# 결과 디코딩 및 출력
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

이 예제는 Mistral-7B-Instruct 모델을 로드하고 `.generate()` 메서드를 사용하여 자동 회귀 텍스트 생성을 처리합니다. 출력은 프롬프트에 대한 응답일 것입니다. `.generate` 메서드에는 디코딩(그리디, 빔 서치, 샘플링 등)을 위한 다양한 매개변수가 있습니다.



### HuggingFace Hub 라이브러리 상세


Hugging Face Hub 라이브러리(`huggingface_hub`)는 Hugging Face 저장소와 상호작용하기 위한 파이썬 인터페이스를 제공하는 low-level 도구입니다. Transformers 라이브러리가 내부적으로 사용하지만, 독립적으로도 사용할 수 있으며 모든 유형의 파일을 관리할 수 있습니다.

#### 1. 인증

Hub와 상호작용하기 위해서는 사용자 인증이 필요합니다:

```python
# 명령줄에서 로그인 (토큰을 로컬에 저장)
!huggingface-cli login

# 또는 파이썬 코드에서 로그인
from huggingface_hub import login
login(token="hf_...")  # 개인 액세스 토큰
```

토큰은 [Hugging Face 설정](https://huggingface.co/settings/tokens)에서 생성할 수 있습니다.

#### 2. 파일 다운로드

특정 모델의 파일만 다운로드할 때 유용합니다:

```python
from huggingface_hub import hf_hub_download

# 특정 파일만 다운로드
config_file = hf_hub_download(
    repo_id="google/mt5-base", 
    filename="config.json"
)
```

#### 3. 파일 업로드 및 저장소 관리

```python
from huggingface_hub import HfApi, upload_file

# API 인스턴스 생성
api = HfApi()

# 새 저장소 생성
api.create_repo(repo_id="username/my-new-model", private=True)

# 단일 파일 업로드
upload_file(
    path_or_fileobj="./model.pt",  # 로컬 파일 경로
    path_in_repo="model.pt",       # 저장소 내 경로
    repo_id="username/my-new-model"
)

# 폴더 전체 업로드
api.upload_folder(
    folder_path="./model_files",
    repo_id="username/my-new-model",
    repo_type="model"
)
```

#### 4. Git과 Repository 클래스 활용

Hub 저장소는 Git 저장소이므로 Git 명령어로 직접 작업할 수도 있습니다:

```python
from huggingface_hub import Repository

# 저장소 로컬에 클론
repo = Repository(
    local_dir="./my-model-dir", 
    clone_from="username/my-model"
)

# 파일 추가/수정 후 커밋 및 푸시
with open("./my-model-dir/README.md", "w") as f:
    f.write("# 내 모델에 대한 설명\n\n이 모델은...")

repo.git_add("README.md")
repo.git_commit("README 업데이트")
repo.git_push()
```

#### 5. 모델 카드 및 메타데이터 관리

```python
from huggingface_hub import ModelCard, CardData

# 모델 카드 생성 또는 업데이트
card_data = CardData(
    language="ko",
    license="mit",
    tags=["text-classification", "korean"],
    datasets=["klue"],
    metrics=[{"name": "accuracy", "value": 0.92}]
)

card = ModelCard.from_template(
    card_data=card_data,
    template_path="modelcard_template.md"  # 옵션: 커스텀 템플릿
)

card.push_to_hub("username/my-model")
```

#### 6. Trainer와의 통합

Transformers `Trainer`를 사용한다면, 훈련이 끝난 후 한 줄로 모델과 메타데이터를 저장소에 업로드할 수 있습니다:

```python
# 훈련 후 저장소에 업로드
trainer.push_to_hub()
```

이 명령은 모델과 토크나이저를 저장하고, 학습 로그와 하이퍼파라미터를 포함한 모델 카드를 자동으로 생성하여 업로드합니다.



### Datasets 라이브러리 상세

🤗 Datasets 라이브러리는 데이터을 효율적으로 로드, 처리, 공유하기 위한 도구입니다. 대규모 데이터셋을 메모리에 맞게 스트리밍하거나 메모리 매핑하는 기능을 제공하며, Hugging Face Hub와 통합되어 수많은 공개 데이터셋에 쉽게 접근할 수 있습니다.

#### 1. 데이터셋 로드

`load_dataset` 함수를 사용하여 다양한 소스에서 데이터셋을 로드할 수 있습니다:

```python
from datasets import load_dataset

# Hub에서 공개 데이터셋 로드
squad_dataset = load_dataset("squad")  # SQuAD 질의응답 데이터셋

# 특정 사용자의 데이터셋 로드
custom_dataset = load_dataset("username/my_dataset")

# 로컬 파일에서 로드
csv_dataset = load_dataset("csv", data_files="my_data.csv")
json_dataset = load_dataset("json", data_files="data.jsonl")

# 여러 파일 또는 분할 지정
multi_dataset = load_dataset("csv", data_files={
    "train": "train_data.csv",
    "validation": "val_data.csv"
})
```

Datasets는 CSV, JSON, Parquet, Text, Image 등 다양한 파일 형식을 지원합니다. 지원되지 않는 형식의 경우, 커스텀 로더 스크립트를 작성하거나 파이썬에서 직접 데이터를 로드하여 Dataset 객체를 만들 수 있습니다.

#### 2. Dataset 객체 구조

데이터셋을 로드하면 일반적으로 `Dataset` 객체(단일 분할) 또는 `DatasetDict`(여러 분할)를 얻게 됩니다:

```python
# Dataset 구조 확인
dataset = load_dataset("glue", "sst2")
print(type(dataset))  # <class 'datasets.dataset_dict.DatasetDict'>
print(dataset.keys())  # dict_keys(['train', 'validation', 'test'])

# 특정 분할 접근
train_dataset = dataset["train"]
print(len(train_dataset))  # 예: 67349 (항목 수)
print(train_dataset.features)  # 컬럼 구조 확인

# 개별 항목 접근
print(train_dataset[0])  # 첫 번째 항목 (딕셔너리 형태)
```


#### 3. 데이터셋 처리

Dataset 객체는 데이터 처리를 위한 다양한 메소드를 제공합니다:

##### map 함수로 데이터 변환

```python
from transformers import AutoTokenizer

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# 텍스트 토큰화 함수
def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# 모든 예제에 함수 적용 (병렬 처리)
tokenized_dataset = dataset["train"].map(
    tokenize_function,
    batched=True,  # 배치 처리로 속도 향상
    num_proc=4     # 병렬 처리 (CPU 코어 수에 따라 조정)
)
```

map 함수는 결과를 자동으로 캐싱하므로, 동일한 처리를 반복할 필요가 없습니다.

##### 필터링과 정렬

```python
# 특정 조건으로 필터링
short_dataset = dataset["train"].filter(lambda x: len(x["sentence"]) < 100)

# 특정 컬럼으로 정렬
sorted_dataset = dataset["train"].sort("label")

# 데이터셋 섞기
shuffled_dataset = dataset["train"].shuffle(seed=42)

# 분할
train_test = dataset["train"].train_test_split(test_size=0.2)
# 결과: DatasetDict({'train': Dataset(...), 'test': Dataset(...)})
```

#### 4. 메모리 관리 및 스트리밍

대규모 데이터셋을 처리할 때 RAM 사용량을 관리하는 몇 가지 방법이 있습니다:

```python
# 스트리밍 모드로 데이터셋 로드 (전체를 메모리에 로드하지 않음)
streamed_dataset = load_dataset("c4", "ko", split="train", streaming=True)

# 이터레이터로 사용
for example in streamed_dataset:
    # 한 번에 하나의 예제만 메모리에 로드
    process_example(example)
    
# 스트리밍 데이터셋의 일부만 확인
for i, example in enumerate(streamed_dataset):
    if i >= 5:  # 처음 5개만 확인
        break
    print(example)
```

스트리밍 데이터셋에도 `.map()`, `.filter()` 등을 적용할 수 있으며, 이 경우 변환이 지연 처리(lazy processing)됩니다.

#### 5. Transformers와 통합

Datasets는 Transformers 라이브러리와 원활하게 통합됩니다:

```python
# PyTorch 텐서 형식으로 설정
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Trainer에 직접 전달
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base")
training_args = TrainingArguments(output_dir="./results", num_train_epochs=3)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"]
)

trainer.train()
```

#### 6. Hub와 데이터셋 공유

처리한 데이터셋을 Hugging Face Hub에 공유할 수 있습니다:

```python
# 데이터셋을 Hub에 업로드
processed_dataset.push_to_hub("username/my_processed_dataset")

# 메타데이터 및 카드 정보 포함
processed_dataset.push_to_hub(
    "username/my_processed_dataset",
    repository_url="https://huggingface.co/datasets/username/my_processed_dataset",
    commit_message="Add processed Korean sentiment dataset"
)
```

데이터셋을 공유하면 다른 사용자들이 `load_dataset("username/my_processed_dataset")`으로 쉽게 이용할 수 있습니다.

#### 8. 대규모 데이터셋 작업 팁

대규모 데이터셋 작업 시 추천되는 방법들:

- 메모리 맵핑: Datasets는 기본적으로 메모리 맵핑을 사용하여 전체 데이터셋을 RAM에 로드하지 않습니다.
- 컬럼 선택: 필요한 컬럼만 선택하여 메모리 사용량 감소 (`dataset.select_columns(["text", "label"])`)
- 배치 처리: `.map(batch_size=1000)`으로 대규모 변환을 효율적으로 처리
- 캐싱 활용: 기본적으로 `.map()` 결과는 캐시되지만, 필요하면 `load_from_cache_file=False`로 비활성화 가능
- 분산 처리: `num_proc` 인자로 멀티프로세싱 활용

Datasets 라이브러리는 대규모 데이터셋도 효율적으로 처리할 수 있게 설계되어, 모델 훈련 및 평가를 위한 데이터 준비 과정을 크게 간소화합니다.



### Hugging Face Spaces (Gradio / Streamlit 앱 호스팅)

Spaces는 Hugging Face의 라이브 머신러닝 데모 및 애플리케이션을 호스팅하기 위한 솔루션입니다. Space는 본질적으로 웹 앱(주로 Gradio 또는 Streamlit 사용)을 실행하고 지속적인 URL을 통해 다른 사용자와 공유할 수 있는 샌드박스 환경입니다.

#### 1. Space 생성 및 공유

- Space는 Git 저장소를 기반으로 합니다
- Hub에서 새 Space를 만들고 이름 선택, 공개/비공개 설정, SDK 유형을 선택합니다
- 코드를 푸시하면 Space가 자동으로 빌드되고 실행됩니다

#### 2. 개발 도구
- **Gradio**: 모델용 웹 인터페이스를 쉽게 만들 수 있는 Python 라이브러리입니다
  - 함수(입력 텍스트를 받아 출력 텍스트를 반환하는 예측 함수 등)를 정의하면 Gradio가 텍스트 박스, 버튼 등과 함께 인터페이스를 생성합니다
- **Streamlit**: Python 스크립트로 작은 웹 앱을 작성하는 것과 더 유사합니다
  - 그래프, 위젯 등에도 직관적인 인터페이스를 제공합니다, 요즘 가장 간단하게 파이썬으로 웹 앱을 만드는 라이브러리로 유명합니다. 

#### 3. 하드웨어 및 스케일링
- **기본 사양**: 무료 CPU 전용 컨테이너(16GB RAM)
  - 작은 모델이나 추론 API를 호출하는 데모에 적합
- **GPU 업그레이드 (유료)**: 대형 모델 호스팅 시 필요
  - T4, A10G, A100 등 다양한 티어 제공
  - 시간당 과금 방식

#### 4. Spaces 활용 사례
- OpenLLM Learderboard











