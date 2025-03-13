# 벤치마크 및 평가 방법론

LLM을 잘 만들었는지 궁금하다면 평가를 해야겠죠.  

가장 쉬운 방법은, 

- 사람들한테 두 가지 모델에서 나온 응답을 보여주고 뭐가 더 나은지 투표를 하게 하는 것입니다.

사람이 투표를 하면 자동 평가가 안되는 것이 문제죠. 다음 방법은 

-  그냥 퀴즈내고 잘 맞추는지 보는 것입니다.


단답형이나 객관식 답변은 단순하게 채점이 가능한데요, 다음 문제는 서술형 평가죠. 요즘은 서술형 답변에 대한 평가도 LLM 에 맡깁니다.  
이를 LLM as Judge 라고 부르는데요, 이건 따로 다음 문서에서 다루겠습니다. 


## 1. LMSYS Chatbot Arena

[Chatbot Arena Leaderboard](https://lmarena.ai/?leaderboard) 에 가시면 현 시점에 가장 많은 선택을 받은 모델들의 순위를 보실 수 있습니다. 
이 글을 작성하고 있는 25.03.10 기준으로는 Grok3 가 1등, 1점차이로 GPT-4.5가 2등이네요.

제가 개인적으로 어떤 모델이 현재 가장 좋을지 궁금할 때 항상 보는 곳입니다. 

<div style="text-align: center;">
  <img src="../../rscs/arena_leaderboard.png" alt="lmarena">
  <p><em>LMSYS Chatbot Arena Leaderboard (25.03.10)</em></p>
</div>

---

누구나 바로 테스트하고 투표 해보실 수 있습니다. 바로 한번 해봤습니다. 
Model B 가 줏대가 있군요. 더 마음에 드는 모델을 투표하면 가려진 이름이 나옵니다. 

<div style="text-align: center;">
  <img src="../../rscs/arena_battle.png" alt="lmarena_battle">
  <p><em>LMSYS Chatbot Arena Battle</em></p>
</div>

Model A 는 claude sonnet 3.7 thinking 이었고, Model B 는 Gemini 2.0 flash lite 이었습니다. 


---

LMSYS Arena 가 전반적인 모델의 성능에 대해 가장 사람들이 신뢰하는 평가 방법인 것 같아요. 물론, 코딩 능력이나 분야에 따라서, 용도에 따라서 다 다를 수 있곘지만요.
블라인드 페어와이즈 테스트, Elo 레이팅 시스템, 수많은 데이터 수집 등 신뢰할 만한 요소가 잘 갖춰진 것 같습니다.  

문제는 시스템에 등록되어 사람들의 투표를 기다려야 하는데요, 이를 해결하기 위해 [Arena Hard Auto](https://github.com/lmarena/arena-hard-auto) 라는 아레나 모사(?) 벤치도 있습니다. 벤치마크 평가는 이어서 보겠습니다. 


## 2. 벤치마크 테스트

LLM 을 평가하기 위한 다양한 벤치마크가 있습니다.  
제일 유명한 것은 아무래도 [MMLU (Massive Multitask Language Understanding)](https://huggingface.co/datasets/cais/mmlu) 겠죠. 한 번쯤은 들어보셨을 텐데요. 57개 과목에 걸친 지식 테스트로, 대학 수준의 문제들을 포함합니다.  

수 많은 문제 중 하나만 발췌해 왔습니다. 

| 문제 | 과목 | 보기 | 정답 |
|------|------|------|------|
| Which of these sets of logic gates are designated as universal gates? | electrical_engineering | A. NOR, NAND.<br>B. XOR, NOR, NAND.<br>C. OR, NOT, AND.<br>D. NOR, NAND, XNOR. | A |

이 처럼 보기가 4개가 있는 객관식이라 평가하기가 아주 쉽습니다. 

또 다른 유명한 벤치들은 다음과 같습니다. 

- **HumanEval/MBPP**: 코딩 능력을 평가하는 벤치마크로, 함수 구현 문제를 제시하고 실행 가능한 코드를 생성하는지 확인합니다. HumanEval은 OpenAI가 개발한 164개의 프로그래밍 문제로 구성되며, MBPP(Mostly Basic Python Problems)는 974개의 기본 파이썬 프로그래밍 문제를 포함합니다.
- **GSM8K/MATH**: 수학 문제 해결 능력을 평가하는 벤치마크입니다. GSM8K는 8,000개의 초중등 수준 수학 문제를, MATH는 대학 입시 수준의 12,500개 고난도 수학 문제를 포함합니다. 단계별 추론 능력을 측정하는 데 효과적입니다.
- **TruthfulQA**: 모델이 사실에 기반한 정확한 정보를 제공하는지 평가합니다. 일반적인 오해, 미신, 잘못된 믿음에 관한 질문을 통해 모델의 사실성과 정직성을 테스트합니다.
- **ARC (AI2 Reasoning Challenge)**: 초등학교 수준의 과학 문제를 통해 추론 능력을 평가합니다. 쉬운 세트와 어려운 세트로 나뉘며, 특히 어려운 세트는 단순 패턴 매칭으로 해결할 수 없는 문제들로 구성되어 있습니다.
- **MMLU-Pro**: MMLU의 확장 버전으로, 더 어려운 전문 지식 문제를 포함합니다.
- **SWE-bench**: 실제 GitHub 이슈를 기반으로 한 소프트웨어 엔지니어링 능력 평가 벤치마크입니다.
- **SimpleQA**: 일반 지식을 테스트하는 질의응답 벤치마크로, 모델의 기본적인 사실 지식을 평가합니다.

다양한 벤치들은 워낙 많기 때문에 이정도만 소개하고 넘어가겠습니다. 

### 자동 평가 도구

#### EleutherAI의 [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness)

아래와 같이 CLI로 아주 쉽게 자동화된 평가가 가능합니다. 

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6B \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

- **모듈식 태스크 플러그인**: 다양한 벤치마크를 일관된 방식으로 평가할 수 있는 통합 프레임워크입니다.
- **다양한 평가 방식 지원**: 제로샷, 원샷, 퓨샷 등 다양한 평가 방식을 지원합니다.
- **다양한 메트릭**: 정확도, F1 점수, 퍼플렉시티 등 태스크에 적합한 메트릭을 제공합니다.
- **재현성**: 동일한 모델과 태스크 설정으로 항상 같은 결과를 얻을 수 있도록 설계되었습니다.

평가 방식에 대해선 자세한 설명이 많이 필요해서 따로 다루도록 하겠습니다.


[lm_eval tasks](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md) 를 보시면 지원하는 다양한 벤치마크들을 보실 수 있습니다. 아주 아주 아주 많이 있습니다.

> 꼭 쓰세요!


## 한국어 LLM 평가 방법

한국어를 위한 벤치마크도 당연히 꽤 있습니다. 

앞서 이야기한 [lm_eval tasks](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/README.md) 에도 한국어 벤치마크들이 있죠. 

최근 카카오에서 한국어를 타겟으로 발표한 모델 [Kanana](https://huggingface.co/kakaocorp/kanana-nano-2.1b-instruct)의 평가 결과를 보겠습니다. 

| 모델 | MMLU | KMMLU | HAE-RAE | HumanEval+ | MBPP+ | GSM8K | MATH |
|------|------|-------|---------|------------|-------|-------|------|
| Kanana-Flag-32.5b | 81.08 | **64.19** | **68.18** | 77.44 | 69.84 | 90.83 | 57.82 |
| Qwen2.5-32b | 84.40 | 59.37 | 48.30 | 82.32 | 71.96 | 95.30 | 81.90 |
| Gemma-2-27b | 78.01 | 49.98 | 46.02 | 70.12 | 70.90 | 91.05 | 53.80 |
| EXAONE-3.5-32b | 78.30 | 55.44 | 52.27 | 78.66 | 70.90 | 93.56 | 76.80 |
| Aya-Expanse-32b | 74.49 | 42.35 | 51.14 | 64.63 | 65.61 | 75.06 | 42.82 |

이 표에서 볼 수 있듯이, Kanana 가 가장 뛰어난 두 개의 벤치 KMMLU 와 HAE-RAE 가 한국어 기반 벤치마크 입니다.  

제가 뽑은 벤치마크들은 다음과 같습니다. 한국어 사용자 분들이 가장 많이 사용하시던 것들을 기준으로 선정했습니다.  
영어 벤치마크를 한국어 기반으로 적용시킨 벤치들이 가장 일반적이고, HAE-RAE 같은 경우는 "한국어" 라기보다 "한국" 문화 벤치 라고 보는 것이 좋겠습니다.  

- **KMMLU**: 한국어로 재구성된 MMLU로, 45개 카테고리의 전문 지식을 평가
- **LogicKor**: LMSYS의 MT-Bench를 한국어로 적응한 다중 턴 대화 벤치마크
- **Open Ko-LLM Leaderboard**: 한국어 LLM을 위한 종합 랭킹 시스템
- **HAE-RAE Bench**: 한국 문화 지식 QA 벤치마크

## 3. 벤치마크의 실효성에 대한 논의

> 벤치마크 성적이 좋은 것이 정말 좋은 모델일까요 ?? 

이 부분에 대해 많은 분들이 의문을 가집니다. 많이들 느끼셨을 거에요. 분명 벤치 숫자는 좋은데 왜 실 체감은 별로지...?  
제 의견으로는 **벤치에 overfit 되어 있어서** 라고 생각합니다.   

MMLU 와 같은 지식은 묻는 벤치는 당연히 세상의 모든 질문을 커버하지 못 합니다, 이미 문제 정답지가 공개된 이상 기출을 많이 풀어본다면 성능을 올릴 수 있겠죠.  
수학이나 뭐 다른 벤치도 마찬가지에요. LLM 이 한번도 안 풀어본 벤치 문제로 테스트하면 성적표가 많이 달라집니다. 


### Open Ko-LLM Leaderboard 와 LogicKor

위 의문을 잘 보여주는 사례가 있었습니다. 

- [한국어 리더보드는 죽었다 (24.03)](https://arca.live/b/alpaca/102052014)

위 링크를 한번 읽어보시는 걸 추천드립니다.  
요약하면, 실체감과 벤치 점수 (여기서는 Open Ko-LLM Leaderboard) 의 괴리가 크다는 점이고. 그래서 새로 벤치 (여기서는 LogicKor)를 만들어 돌렸더니 실체감과 비슷하더라는 사례입니다. 영어권에서도 당연히 비슷한 문제가 제기가 되고 있습니다.
[Pretraining on the Test Set Is All You Need](https://arxiv.org/abs/2309.08632) 페이퍼는 테스트 벤치를 직접 학습하는 것 (기출 문제를 풀어보고 시험 보는 것) 에 대한 비판입니다.   

- [LogicKor 운영 중지](https://github.com/instructkr/LogicKor)

새롭게 만들어진 벤치도 당연히 공개가 되고 나면 이를 학습하고 새로운 모델들에게 뚫리기 마련입니다. 비슷한 맥락으로 운영은 중지가 되었습니다. 

- [2025 수능 언어영역 벤치](https://github.com/Marker-Inc-Korea/Korean-SAT-LLM-Leaderboard/)

비슷한 맥락으로 Marker AI 에서 만들어주신 24년 말 실시된 수능 언어영역에 대한 벤치와 그 결과입니다. 새로 나온 문제들은 아직 학습한 LLM 이 없기 떄문에 정당한 평가 요소로 사용될 수 있었습니다.   


참고로 LogicKor 를 만드신 maywell 선생님은 고등학생 이시네요?! 대단하시군요 크흠...



## 최근 발전 및 추가 벤치마크

최근에는 단순 지식과 능력을 넘어서 보다 다양한 평가요소들이 떠오르고 있습니다.  
Reasoning 이나 AGI, 멀티모달리티, 이런 요소들을 다루고 있습니다.  
GPT4.5는 EQ 를 강조하기도 했죠...


- **HELM (Holistic Evaluation of Language Models)**: 정확성, 견고성, 공정성 등 다양한 차원에서 LLM을 평가
- **도메인별 벤치마크**:
  - 코드 및 Reasoning: HumanEval-X, APPS, LeetCode 평가
  - 멀티모달: MME, VQAv2
  - 상호작용 에이전트: WebArena, MATH-GPT
  - 안전성 및 정렬: toxicity 평가, 탈옥 테스트

## 참고문헌

1. LMSYS Org, "Chatbot Arena: Benchmarking LLMs in the Wild with Elo Ratings", 2023.
2. LMSYS Org, "From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline", 2024.
3. EleutherAI, "Language Model Evaluation Harness", 2023.
4. Stanford CRFM, "Holistic Evaluation of Language Models (HELM)", 2023.
6. Kim et al., "KMMLU: Korean Massive Multitask Language Understanding", 2023.
7. Lee et al., "Open Ko-LLM Leaderboard: Evaluating Large Language Models in Korean with Ko-H5 Benchmark", 2024.





