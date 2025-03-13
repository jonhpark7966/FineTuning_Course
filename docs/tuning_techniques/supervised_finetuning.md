# Supervised Fine-Tuning (SFT)

## SFT 개요
- SFT(Supervised Fine-Tuning)는 사전학습된 언어 모델을 특정 작업이나 사용자 지시에 맞게 조정하는 기법입니다.
- PreTrained Base 모델이 가진 단순한 "Next Token Prediction" 능력을 넘어, 사용자 질문에 유용하게 답변하는 능력 (Instruction Following) 을 부여합니다.
- 지시-응답 쌍으로 구성된 데이터셋을 사용해 모델이 사람의 의도에 맞는 출력을 생성하도록 훈련합니다.
- 이 외에도 knowledge distillation 이나 style tuning, 최근에는 Reasoning 능력을 향상시키는 등 다양하게 활용됩니다. 

---

혹시라도 SFT 에 대해서 익숙하지 않으시다면 아래 글을 보고 오시는 것을 추천드립니다. 

| 💡 추천 자료 |
|---------|
| [Supervised Fine-Tuning 이해하기](https://sudormrf.run/2025/02/27/supervised-fine-tuning/) - 제가 작성했던 SFT의 기본 개념과 예시를 설명한 글입니다. |


## SFT의 중요성

### 사전학습 모델의 한계 극복
- 사전학습된 베이스 모델은 텍스트 생성 능력은 있지만, 사용자 의도를 정확히 파악하거나 지시를 따르는 능력이 부족합니다.
- SFT는 모델이 사용자 지시를 이해하고 유용한 응답을 생성하도록 행동을 교정합니다.
- 대부분의 사용자 분들이 익숙한 ChatGPT 는 당연히 이미 SFT 된 (그리고 더 많은 개선이 된) 모델을 기반으로 합니다.  

### 실용성 확대
- SFT를 통해 모델은 질문 응답, 요약, 코드 생성 등 다양한 작업을 수행할 수 있게 됩니다.
- 특정 도메인(의료, 법률, 금융 등)에 특화된 응답을 생성하도록 조정할 수 있습니다.

### Alignment
- SFT는 모델이 인간의 가치와 의도에 맞게 행동하도록 Align하는 첫 단계입니다.
- 유해하거나 부적절한 응답을 줄이고, 도움이 되는 정보를 제공하도록 훈련합니다.

## SFT 데이터 준비

### 지시-응답 쌍 구성
- SFT 데이터는 기본적으로 (지시/질문, 기대 응답) 쌍으로 구성됩니다. Instruction 모댈에게 튜닝을 시키기 떄문이죠.  
- 데이터 수집 방법:
    - **인간 작성 데이터**: 전문가나 평가자가 직접 작성한 고품질 응답
        - ex. OpenAI InstructGPT의 13K 데이터셋, [InstructGPT](https://arxiv.org/pdf/2203.02155) 페이퍼에 어떻게 실제로 사람들을 고용해서 데이터를 작성했는지 잘 나와있습니다.
    - **합성 데이터**: 기존 강력한 LLM을 활용해 생성한 지시-응답 쌍 (예: Self-Instruct, Alpaca 데이터셋, UltraChat 등등.)

### 데이터 다양성 확보
- 다양한 유형의 지시를 포함해야 모델의 일반화 능력이 향상됩니다:
  - 질문-답변, 분류, 요약, 번역, 창의적 글쓰기 등 다양한 작업
  - 간단한 질문부터 복잡한 추론이 필요한 질문까지 난이도 변화
  - 여러 주제와 도메인을 포괄하는 질문들

### 품질 중심 데이터 큐레이션
- 데이터 품질이 SFT 성공의 핵심 요소입니다.
- [Meta의 LIMA 연구](https://arxiv.org/pdf/2305.11206) 에 따르면, 1,000개의 고품질 예제만으로도 우수한 성능을 달성할 수 있습니다.
- Alpaca 도 52,000 개로 좋은 효과를 보여줬고, 최근 S1 의 경우도 1,000개의 데이터셋으로 reasoning 모델을 만들었다고 합니다.  
- 품질 기준:
    - 정확성: 사실에 기반한 정보 제공
    - 유용성: 질문에 직접적으로 관련된 도움이 되는 응답
    - 명확성: 이해하기 쉽고 논리적인 구조
    - 안전성: 유해하거나 편향된 내용 배제

## SFT 최적화 전략

### 하이퍼파라미터 설정
- **학습률(Learning Rate)**: 일반적으로 1e-5 ~ 5e-5 범위의 낮은 값 사용
- **배치 크기(Batch Size)**: 메모리 한계 내에서 가능한 크게 설정 (8-32)
- **에포크(Epoch)**: 데이터 양에 따라 조절 (대량 데이터: 1-2 에포크, 소량 데이터: 3-5 에포크)
- **학습 스케줄러**: 선형 또는 코사인 감쇄와 초기 워밍업(3-5%) 적용

### 과적합 방지 기법
- **조기 종료(Early Stopping)**: 검증 손실이 더 이상 개선되지 않을 때 학습 중단
- **가중치 감쇄(Weight Decay)**: 일반적으로 0.01 내외로 설정
- **데이터 증강**: 기존 데이터의 변형을 통해 다양성 확보
- **계층 동결(Layer Freezing)**: 하위 계층은 고정하고 상위 계층만 미세조정

### 효율적 미세조정 기법
- **LoRA(Low-Rank Adaptation)**: 원본 가중치는 고정하고 저랭크 행렬만 학습하여 메모리 효율성 증가
- **QLoRA**: 양자화된 모델에 LoRA 적용하여 더 큰 모델도 적은 메모리로 미세조정 가능
- **Prefix/Prompt Tuning**: 모델 파라미터 대신 입력에 학습 가능한 토큰 추가

### 평가 및 모니터링
- 검증 세트에서 정기적으로 성능 측정 (손실, 정확도, 생성 품질 등)
- 인간 평가를 통한 출력 품질 검증 (유용성, 정확성, 안전성 등)
- 다양한 프롬프트에 대한 일반화 능력 확인

## SFT 구현 예제

아래는 Hugging Face Transformers를 사용한 간단한 SFT 구현 예시입니다:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
# 1. 프리트레인 된 GPT-2 모델과 토크나이저 불러오기
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. 예시용 지도 학습 데이터 정의 (Q&A 형태)
train_pairs = [
    {"prompt": "Q: What is the capital of France?\nA:", "response": " Paris."},
    {"prompt": "Q: Explain the theory of relativity in simple terms.\nA:",
     "response": " It is a physics theory by Einstein that says space and time are linked together..."},
    # ... (추가 데이터)
]

# 3. 데이터셋 전처리: 프롬프트와 답변을 이어붙여 토큰화하고 레이블 생성
train_encodings = {"input_ids": [], "attention_mask": [], "labels": []}
for pair in train_pairs:
    # 프롬프트 + 응답을 하나의 시퀀스로 만들고 토큰화
    text = pair["prompt"] + pair["response"]
    enc = tokenizer(text, truncation=True, max_length=128)
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]
    # 레이블 생성: 프롬프트 부분은 -100으로 마스킹하고, 답변 부분만 학습 대상으로 설정
    # (GPT-2는 causal LM이므로, 프롬프트까지 모두 입력으로 받고 답변 부분만 손실 계산)
    prompt_len = len(tokenizer(pair["prompt"])["input_ids"])
    labels = [-100]*prompt_len + input_ids[prompt_len:]  # 프롬프트 토큰에 대응되는 부분 -100
    # 인코딩 결과 저장
    train_encodings["input_ids"].append(input_ids)
    train_encodings["attention_mask"].append(attn_mask)
    train_encodings["labels"].append(labels)

# 4. Trainer를 활용한 모델 미세조정 설정
training_args = TrainingArguments(
    output_dir="./sft-model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-5,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=10,
    logging_dir="./logs",
    evaluation_strategy="no",   # (예시에서는 검증 생략)
    save_strategy="epoch"
)
trainer = Trainer(model=model, args=training_args,
                  train_dataset=list(range(len(train_encodings["input_ids"]))),  # dummy indices
                  # Trainer를 쓰려면 Dataset 형식 필요. 여기서는 개념 설명을 위해 생략
                  tokenizer=tokenizer, data_collator=tokenizer)
# 실제로는 train_encodings를 HuggingFace Dataset으로 변환하여 trainer.train() 호출
```

## SFT와 다른 학습 방법 비교

### SFT vs 사전학습(Pre-training)
- **사전학습**: 대규모 텍스트 코퍼스에서 다음 토큰 예측을 통해 언어 이해 능력 습득
- **SFT**: 지시-응답 쌍을 통해 특정 작업 수행 및 사용자 의도 이해 능력 향상
- **차이점**: 사전학습은 일반적 언어 능력 개발, SFT는 특정 작업 및 형식에 맞는 응답 생성에 초점

### SFT vs 연속 사전학습(CPT)
- **연속 사전학습**: 추가 코퍼스로 기존 사전학습을 계속하여 지식 확장/업데이트
- **SFT**: 지도 학습을 통해 모델 행동 교정 및 특정 작업 수행 능력 향상
- **차이점**: CPT는 지식 확장에 중점, SFT는 행동 조정에 중점

### SFT vs 강화학습(RL)
- **SFT**: 직접적인 지도 학습으로 정답 예시를 모방하도록 훈련
- **RL**: 보상 함수를 통해 모델이 더 나은 응답을 생성하도록 간접적으로 유도
- **관계**: SFT는 종종 RL의 초기 정책 모델로 사용됨 (예: RLHF에서 SFT 모델이 기반)
- **효과**: 최근 연구에 따르면 고품질 SFT만으로도 RL 수준의 성능을 달성할 수 있음

## SFT의 한계와 고려사항

### 데이터 의존성
- SFT 성능은 훈련 데이터의 품질과 다양성에 크게 의존합니다.
- 편향되거나 부정확한 데이터는 모델의 출력에도 반영됩니다.

### 과적합 위험
- 특히 소량의 데이터로 학습할 때 과적합이 발생할 수 있습니다.
- 모델이 훈련 데이터의 패턴만 암기하고 새로운 입력에 일반화하지 못할 수 있습니다.

### 지식 망각
- SFT 과정에서 모델이 사전학습 단계에서 습득한 일부 지식을 잊을 수 있습니다.
- 특히 공격적인 학습률이나 긴 학습 기간은 이 문제를 악화시킬 수 있습니다.

### 평가의 어려움
- 생성 모델의 출력 품질을 자동으로 평가하기 어렵습니다.
- 인간 평가는 비용이 많이 들고 주관적일 수 있습니다.

# SFT 성공 사례

## Stanford Alpaca: 저비용으로 GPT-3.5 수준 달성
- **모델 기반**: Meta의 LLaMA 7B
- **데이터**: GPT-3.5(text-davinci-003)로 생성한 52,000개의 지시-응답 쌍
- **비용**: 약 $500 (OpenAI API 사용)
- **성과**: 블라인드 평가에서 GPT-3.5와 유사한 성능 달성 (90:89로 근소하게 앞섬)
- **의의**: 소규모 공개 모델도 적절한 SFT로 상용 AI 수준에 근접할 수 있음을 증명

## Vicuna: 실제 사용자 대화로 학습한 오픈소스 챗봇
- **모델 기반**: LLaMA 13B
- **데이터**: ShareGPT에서 수집한 70,000건의 사용자-ChatGPT 대화
- **비용**: 약 $300
- **성과**: GPT-4 평가에서 ChatGPT 품질의 90% 이상 달성
- **의의**: 실제 사용자 대화 데이터로 SFT하여 다중 턴 대화에 강한 오픈소스 챗봇 개발

## Microsoft Orca: GPT-4 추론 과정 모방
- **모델 기반**: 13B 규모 모델
- **데이터**: GPT-4가 생성한 단계별 설명(trace)과 사고 과정
- **성과**:
  - 동일 크기 모델(Vicuna-13B) 대비 복잡한 추론 벤치마크에서 2배 이상 성능 향상
  - Big-Bench Hard(BBH)에서 ChatGPT와 거의 대등한 성능
  - AGIEval 벤치마크에서 시스템 프롬프트 최적화된 ChatGPT에 근접
- **의의**: 대형 모델의 사고 과정을 모방 학습하는 방식이 작은 모델의 추론 능력 향상에 효과적임을 입증

## DeepSeek R1 지식 증류: 초거대 모델의 추론 능력 전수
- **접근법**: 초거대 언어모델(DeepSeek R1)을 교사로 활용한 지식 증류
- **특징**: 복잡한 문제에 대한 단계별 추론 과정(Chain-of-Thought)을 포함한 합성 데이터 생성
- **성과**:
  - LLaMA 기반 8B 모델 실험에서 최종 답만 학습 시 29%, 인간 전문가 풀이과정 학습 시 68%, DeepSeek R1 풀이과정 학습 시 87%의 정확도 달성
  - 인간 전문가가 작성한 해설보다 DeepSeek R1이 생성한 해설 데이터가 더 효과적
- **의의**: 대형 모델의 지식을 작은 모델에 효과적으로 이전하는 새로운 SFT 방향 제시

## 도메인 특화 SFT 성공 사례

### WizardCoder: 코드 생성 특화 모델
- **모델 기반**: 다양한 크기의 코드 언어 모델(15B, 34B 등)
- **데이터**: 고품질 문제-해결 예시를 증강(Evol-Instruct)한 데이터
- **성과**:
  - WizardCoder-15B: HumanEval 벤치마크에서 57.3% 정답률(pass@1)
  - WizardCoder-34B: 73% 이상의 정답률로 GPT-4(2023년 3월 버전) 성능 초과
  - WizardCoder-33B v1.1: 79.9% 정답률로 ChatGPT-3.5(72.6%) 능가
- **의의**: 제한된 파라미터로도 최적의 데이터와 기법을 적용하면 상용 AI 수준의 코딩 성능 달성 가능

### WizardMath: 수학 문제 해결 특화 모델
- **모델 기반**: Meta의 LLaMA-2 모델들
- **데이터**: 고난도 수학 문제를 단계별로 풀이하는 데이터
- **기법**: 강화 학습(RL)을 병합한 독자 기법(RLEIF)
- **성과**:
  - WizardMath-13B: MATH 벤치마크에서 LLaMA2-70B보다 9.2%p 높은 점수
  - WizardMath-70B: ChatGPT-3.5, Claude Instant 등을 뛰어넘는 수학 성능
  - WizardMath-7B: 같은 7B급에서 독보적인 수학 실력(GSM8K 83.2% 등)
- **의의**: 수학적 사고 과정을 집중적으로 훈련시키는 SFT가 모델의 문제 해결 능력을 크게 향상시킬 수 있음을 증명

## 참고문헌

1. Stanford Alpaca: ["Alpaca: A Strong, Replicable Instruction-Following Model"](https://crfm.stanford.edu/2023/03/13/alpaca.html), Stanford CRFM, 2023.

2. Vicuna: ["Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality"](https://lmsys.org/blog/2023-03-30-vicuna/), LMSYS Org, 2023.

3. Microsoft Orca: ["Orca: Progressive Learning from Complex Explanation Traces of GPT-4"](https://www.microsoft.com/en-us/research/publication/orca-progressive-learning-from-complex-explanation-traces-of-gpt-4/), Microsoft Research, 2023.

4. QLoRA & Guanaco: ["QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314), Dettmers et al., 2023.

5. DeepSeek R1: "Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", DeepSeek AI, 2024.

6. DeepSeek R1 Distillation: ["Distillation with Reasoning: Can DeepSeek R1 Teach Better Than Humans?"](https://fireworks.ai/blog/deepseek-r1-distillation-reasoning), Fireworks AI, 2024.

7. WizardCoder: ["WizardCoder: Empowering Code Large Language Models with Evol-Instruct"](https://github.com/nlpxucan/WizardLM), Xu et al., 2023.

8. WizardMath: ["WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct"](https://openreview.net/forum?id=mMPMHWOdOy), Luo et al., 2023.

9. ["What is supervised fine-tuning in LLMs? Unveiling the process"](https://nebius.com/blog/posts/fine-tuning/supervised-fine-tuning), Nebius, 2023.

10. ["LLM continuous self-instruct fine-tuning framework powered by a compound AI system on Amazon SageMaker"](https://aws.amazon.com/blogs/machine-learning/llm-continuous-self-instruct-fine-tuning-framework-powered-by-a-compound-ai-system-on-amazon-sagemaker/), AWS Machine Learning Blog, 2023.