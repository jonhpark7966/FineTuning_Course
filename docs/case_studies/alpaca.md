# Alpaca - Llama Instruction Tuning

<div style="text-align: center;">
  <img src="../../rscs/alpaca_logo.png" alt="alpaca_logo">
</div>

## Alpaca 프로젝트 개요 - 2023.3

Alpaca는 Stanford 대학에서 개발한 경량 명령어-따르기(instruction-following) 언어 모델입니다. Meta의 LLaMA 7B를 기반으로 52K개의 지시-응답 데이터에 대해 파인튜닝되었으며, GPT-3.5 수준의 성능을 보이면서도 재현 비용이 매우 낮다는 특징이 있습니다.

- **개발 배경**: Closed Source LLM (ex. 당시에는 GPT-3.5)에 필적하면서도 학계 연구자들이 쉽게 활용할 수 있는 오픈소스 대안을 제공하고자 했습니다.
- **저비용 접근**: 전체 파인튜닝 비용이 $600 미만으로, 데이터 생성에 약 $500, 모델 학습에 약 $100이 소요되었습니다.
- **진정한 오픈소스**: 모델 코드, 데이터 (데이터 생성 코드), 학습 코드 모두 공개했습니다. 


<div style="text-align: center;">
  <img src="../../rscs/alpaca_main.jpg" alt="alpaca_main">
</div>


## 데이터 및 방법론

### 1. 데이터 생성 과정
- 사람이 직접 175개의 시드(instruction-output 쌍) 를 준비하고
- OpenAI GPT-3.5 (text-davinci-003) 모델을 활용한 [self-instruct](https://arxiv.org/abs/2212.10560) 기법으로 52K 규모의 Instruction-Output 쌍 데이터를 생성합니다.
- 데이터 생성 비용 <$500 (OpenAI API 사용)

### 2. 모델 튜닝 과정
- Meta의 LLaMA 7B 모델을 기반으로 SFT (Supervised Fine-Tuning) 를 합니다. 
- 80GB A100 GPU 8대로 3시간 소요 (약 $100 비용) 되었다고 하더군요. 
- HuggingFace 프레임워크 활용했고, 코드도 공개했습니다. [학습 코드](https://github.com/tatsu-lab/stanford_alpaca#fine-tuning) 

## 결과 & 의의

### 성능
- 싱글턴 지시 수행 평가에서 OpenAI text-davinci-003와 대등한 수준 달성했습니다!
- 179개 비교 중 Alpaca 90건, GPT-3.5 89건으로 비슷한 성능을 보였습니다. 
- 7B 수준의 작은 모델로도 실사용 수준의 성능 달성 가능성 입증한 것이 가장 큰 의의라고 보시면 되겠습니다. 

### 데이터 효율성
- 52k 개의 상대적으로 적은 지도 데이터로도 큰 성능 개선을 이루었습니다
- GPT-3.5로 생성된 고품질 데이터의 효과성을 입증했습니다
- 비용 대비 높은 성능 향상을 달성했습니다

### 라이선스 및 제한 사항
- 비영리 연구 목적으로만 공개되었습니다 (LLaMA 라이선스 제약)
- 환각(hallucination) 현상이 존재합니다

## 파생 모델 및 영향

### 다양한 언어 버전
- KoAlpaca: 한국어 명령어 대응을 위한 파생 모델이 개발되었습니다
- 각 언어권별 최적화 모델들이 등장했습니다

### 오픈소스 생태계 영향
- Meta의 LLaMA 유출 이후 다수의 파생 LLM이 등장했습니다
- Vicuna-13B 등 성능 개선된 후속 모델들이 출현했습니다
- 오픈소스 LLM의 민주화를 촉진했습니다

보다 자세한 내용이 궁금하시면, LLM 배경지식 문서를 참조해 주세요.

참고 자료: [Stanford CRFM - Alpaca 프로젝트](https://crfm.stanford.edu/2023/03/13/alpaca.html) 