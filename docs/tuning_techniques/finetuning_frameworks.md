# Supervised Fine-Tuning (SFT)

## SFT 개요
- SFT의 목적과 중요성
- 입력-출력 매핑을 통한 학습

## SFT 데이터 준비
- 지시-응답 쌍 구성
- 다양한 지시 형태 포함하기
- 품질 중심 데이터 큐레이션

## SFT 최적화 전략
- 과적합 방지 기법
- 학습률 스케줄링
- 조기 종료 전략 




# 대규모 언어모델(LLM) 지도 파인튜닝: Hugging Face vs. DeepSpeed vs. Unsloth

## 개요 (Introduction)
대규모 언어모델(LLM)의 **지도 학습 기반 파인튜닝**(Supervised Fine-Tuning, SFT)은 사전 학습된 모델을 새로운 데이터셋에 맞춰 미세조정하여 특정 작업 성능이나 응답 품질을 높이는 과정입니다. 특히 **Decoder-Only Transformer** 구조(예: GPT 계열 모델)의 파인튜닝은 주어진 프롬프트에 이어지는 다음 토큰을 예측하도록 모델을 학습시키는 형태로 이루어집니다. 최근 2년간 LLM 파인튜닝 분야에서는 **모델 크기에 비해 한정된 자원으로도 효율적으로 학습**할 수 있는 다양한 기법과 도구들이 등장했습니다. 본 보고서에서는 Hugging Face 생태계, Microsoft DeepSpeed, 그리고 최신 커뮤니티 툴인 Unsloth를 활용한 **지도 파인튜닝 방법**을 비교합니다. 또한 각 접근법의 **특징과 장점**, **최신 연구 동향**, **실무 적용 예제 코드**, **성능 및 효율 평가 기준**, **Decoder-Only 최적화 기법** 등을 정리합니다.

## Hugging Face 기반 LLM 파인튜닝
**Hugging Face**의 Transformers 라이브러리는 방대한 사전학습 모델 저장소와 편리한 API를 제공하여 LLM 파인튜닝을 손쉽게 시작할 수 있게 해줍니다. PyTorch 기반으로 구현된 `Trainer` 클래스 또는 **🤗 Accelerate**를 통해 단일 GPU부터 분산 GPU까지 **손쉬운 학습 스크립트 구성**이 가능합니다. Hugging Face의 주요 강점은 **광범위한 모델 지원**과 **커뮤니티 중심의 신속한 개선**입니다. 예를 들어, 2022년 발표된 **LoRA**(Low-Rank Adaptation) 기 ([Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes#:~:text=More%20specifically%2C%20QLoRA%20uses%204,in%20the%20original%20LoRA%20paper))】을 빠르게 PEFT 라이브러리에 통합하고, 2023년 등장한 **QLoRA** 방법론도 곧바로 지원하였습니 ([Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes#:~:text=LLMs%20are%20known%20to%20be,the%20QLoRA%20paper%20by%20Dettmers))】. 이를 통해 사용자는 최소한의 코드 변경만으로 최신 연구 성과를 실습에 적용할 수 있습니다.

Hugging Face는 **메모리 최적화**를 위해 8-bit 및 4-bit 양자화(qunatization)를 지원합니다. 예를 들어 `transformers`에서 `from_pretrained` 호출 시 `load_in_4bit=True`로 설정하면, 사전학습된 모델 가중치를 4비트 정밀도로 불러올 수 있습니 ([Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes#:~:text=As%20a%20quickstart%2C%20load%20a,0)) ([Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes#:~:text=The%20basic%20way%20to%20load,that%20will%20be%20automatically%20inferred))】. 이렇게 하면 모델 메모리 사용량을 크게 줄일 수 있어, 비교적 **적은 GPU 메모리로도 대형 모델을 다룰 수 있게 됩니다* ([Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes#:~:text=In%20few%20words%2C%20QLoRA%20reduces,on%20a%20single%2046GB%20GPU))】. 아래 예시는 OPT-350M 모델을 4비트로 불러오는 코드입니다:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-350m",
    load_in_4bit=True,  # 4비트 양자화 로드
    device_map="auto"   # 가용 GPU 자동할당
)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
```

또한 Hugging Face PEFT 라이브러리를 사용하면 **LoRA 어댑터**를 손쉽게 적용할 수 있습니다. LoRA는 모델의 모든 가중치를 미세조정하는 대신, **일부 매트릭스에 소규모의 학습가능한 저랭크 행렬**(Adapters)을 추가하여 학습하는 방법입니 ([Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes#:~:text=More%20specifically%2C%20QLoRA%20uses%204,in%20the%20original%20LoRA%20paper))】. Hugging Face는 `peft.LoraConfig`와 `get_peft_model` 등을 통해 기존 모델에 LoRA 모듈을 삽입할 수 있는 API를 제공합니다. LoRA를 사용하면 파인튜닝시 **메모리와 연산량을 크게 절감**하면서도 원래 모델의 성능을 거의 유지할 수 있습니다. 2023년 제안된 **QLoRA**는 이를 한 단계 발전시켜 **사전학습 모델을 4-bit로 고정**하고 LoRA로만 업데이트를 수행함으로써, **65억~130억급 모델도 단일 GPU로 미세조정 가능**하게 만들었습니 ([Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes#:~:text=In%20few%20words%2C%20QLoRA%20reduces,on%20a%20single%2046GB%20GPU))】. 실제로 QLoRA를 통해 **65B 파라미터 모델을 48GB VRAM의 단일 GPU에서 풀 16비트 파인튜닝과 동등한 성능으로 학습**하는 데 성공했습니 ([Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes#:~:text=enough%20to%20finetune%20a%2065B,we%20name%20Guanaco%2C%20outperforms%20all)) ([Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes#:~:text=QLoRA%20tuning%20is%20shown%20to,the%20power%20of%20QLoRA%20tuning))】. 이는 **GPU 1대에서 780GB 메모리가 필요했던 작업을 48GB로 줄인 성과**로, 대규모 모델 파인튜닝의 **접근성을 혁신적으로 향상**시켰습니 ([[2305.14314] QLoRA: Efficient Finetuning of Quantized LLMs - ar5iv](https://ar5iv.labs.arxiv.org/html/2305.14314#:~:text=ar5iv%20ar5iv,finetunable%20on%20a%20single%20GPU))】. QLoRA의 핵심 아이디어는 **NF4 (4-bit NormalFloat) 양자화**와 **이중 양자화(Double Quantization)**, 그리고 **Paged Optimizer** 등을 도입하여 성능 저하 없이 메모리를 극단적으로 아낀 것입니 ([[2305.14314] QLoRA: Efficient Finetuning of Quantized LLMs](https://ar5iv.org/abs/2305.14314#:~:text=We%20present%20QLoRA%2C%20an%20efficient,new%20data%20type%20that%20is)) ([[2305.14314] QLoRA: Efficient Finetuning of Quantized LLMs](https://ar5iv.org/abs/2305.14314#:~:text=sacrificing%20performance%3A%20%28a%29%204,of))】.

Hugging Face 방법의 장점은 **간편함과 범용성**입니다. 방대한 사전학습 **체크포인트를 Hugging Face Hub에서 즉시 불러와** 활용할 수 있고, **데이터 전처리부터 평가까지 통합된 생태계**(🤗 Datasets 등)를 제공합니다. 특히 **Transformer-Decorder 모델**(예: GPT-2, GPT-3, LLaMA 등)의 **텍스트 생성 태스크**를 위한 파인튜닝 예제가 풍부하며, 학습 loop, 토크나이저, 모델 병렬화 등이 잘 추상화되어 있습니다. 기본 `Trainer`를 사용하는 예시 코드 (예: GPT-2를 텍스트 생성 데이터로 파인튜닝) 는 다음과 같습니다:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    fp16=True,                      # FP16 혼합 정밀도 사용
    logging_steps=100,
    save_steps=500,
    deepspeed=None                  # (DeepSpeed 사용시 설정 파일 경로 지정)
)
trainer = Trainer(model=model, args=training_args, 
                  train_dataset=train_ds, eval_dataset=eval_ds, 
                  data_collator=data_collator)
trainer.train()
```

> **참고:** 상기 코드에서 `deepspeed=None`로 두면 Hugging Face의 기본 Trainer로 학습합니다. DeepSpeed를 사용하려면 `deepspeed="ds_config.json"`처럼 설정 파일을 지정하거나 🤗 Accelerate를 사용합니다 (아래 DeepSpeed 섹션 참고).

Hugging Face 기반 접근의 단점이라면, **아주 큰 모델을 다룰 때는 기본 환경으로는 한계**가 있다는 것입니다. 예를 들어 수십억~수백억 파라미터 모델을 전체 미세조정(full fine-tuning)하려면 멀티 GPU가 필수이며, 이 경우 **DeepSpeed나 FSDP** 등의 보조가 필요할 수 있습니다. 하지만 중소규모 모델이나 LoRA같은 파라미터 효율 기법을 사용한다면 Hugging Face만으로도 충분히 실험이 가능합니다. 결과적으로 Hugging Face는 **연구 개발의 출발점**으로서 최신 기법들을 빠르게 받아들이고 있어, **실무에서도 가장 널리 쓰이는 LLM 파인튜닝 플랫폼**입니다.

## DeepSpeed를 활용한 대규모 모델 파인튜닝
**DeepSpeed**는 마이크로소프트가 개발한 **대규모 분산 학습 최적화 라이브러리**로, 특히 거대 언어모델의 학습을 **속도와 스케일** 측면에서 지원합니다. DeepSpeed의 핵심에는 **ZeRO (Zero Redundancy Optimizer)** 알고리즘이 있 ([DeepSpeed](https://huggingface.co/docs/peft/main/en/accelerate/deepspeed#:~:text=DeepSpeed%20is%20a%20library%20designed,leveraging%20CPU%20resources%20during%20optimization))10】. ZeRO는 데이터 병렬 학습 시 각 GPU에 동일하게 복제되던 **옵티마이저 상태, 그래디언트, 모델 파라미터**를 shard(분할)하여 **GPU들 간에 분산 저장** ([DeepSpeed](https://huggingface.co/docs/peft/main/en/accelerate/deepspeed#:~:text=DeepSpeed%20is%20a%20library%20designed,leveraging%20CPU%20resources%20during%20optimization))10】. 이렇게 하면 중복으로 메모리를 잡아먹는 요소가 사라져, **모델 크기가 커져도 메모리 효율적으로 분산**시킬 수 있습니다. ZeRO는 단계별로 발전되어 **Stage 1**(옵티마이저 상태 분산), **Stage 2**(+ gradient 분산), **Stage 3**(+ 파라미터 자체 분산)으로 구분되며, Stage 숫자가 높을수록 GPU 메모리 부담이 감소 ([DeepSpeed](https://huggingface.co/docs/peft/main/en/accelerate/deepspeed#:~:text=DeepSpeed%20is%20a%20library%20designed,leveraging%20CPU%20resources%20during%20optimization))10】. 특히 **ZeRO-3**는 모든 모델 파라미터를 모든 GPU에 나누어 올려놓고 필요 시 동적으로 불러쓰는 방식으로, **개별 GPU에는 전체 모델의 일부만 상주**하게 됩니다. 이를 통해 예를 들어 **70억~130억 개 파라미터 모델을 단일 또는 소수 GPU에서 학습**시키는 것이 가능해졌 ([ZeRO-Offload - DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/#:~:text=ZeRO,No%20code%20changes%20are%20needed))09】. DeepSpeed 팀의 튜토리얼에 따르면, ZeRO-Offload 기능까지 활용하면 **10억~13억 파라미터 GPT-2 모델도 단일 32GB GPU에서 학습**할 수 있 ([ZeRO-Offload - DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/#:~:text=ZeRO,No%20code%20changes%20are%20needed))09】. 아래는 DeepSpeed의 ZeRO-Offload를 통한 단일 GPU 대용량 모델 학습 사례입니다:

- *“ZeRO-Offload는 옵티마이저 메모리와 연산을 CPU로 오프로드하여, 최대 130억 파라미터에 달하는 큰 모델도 단일 GPU에서 효율적으로 학습할 수 있게 해 ([ZeRO-Offload - DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/#:~:text=ZeRO,No%20code%20changes%20are%20needed))09】.”*

DeepSpeed의 또 다른 강점은 **병렬화와 최적화 전략의 다양성**입니다. 모델 병렬화, 파이프라인 병렬화, mixed precision 연산, Gradient Accumulation 등의 기법을 통합적으로 지원하여 **GPU 여러 대를 최대한 활용**할 수 있습니다. 또한 **CPU Offloading**(ZeRO-Offload)과 **NVMe Offloading**(ZeRO-Infinity)을 통해, GPU 메모리가 부족할 경우 **일부 데이터(예: 모델 가중치나 옵티마이저 상태)를 CPU RAM이나 SSD로 분산**시킴으로써 **사실상 무제한에 가까운 모델 사이즈**까지도 학습을 시도할 수 있 ([ZeRO-Infinity and DeepSpeed: Unlocking unprecedented model scale for deep learning training - Microsoft Research](https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/#:~:text=ZeRO,Infinity%20include)) ([ZeRO-Infinity and DeepSpeed: Unlocking unprecedented model scale for deep learning training - Microsoft Research](https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/#:~:text=,efficiency%20and%20superlinear%20throughput%20scaling))94】. 이러한 극단적 확장성 덕분에 DeepSpeed는 GPT-3(175B) 급 모델 학습이나 수조 개 파라미터 실험처럼 **최첨단 스케일의 연구에 필수적인 도구**로 자리잡았습니다.

DeepSpeed를 실무에 활용하려면 **설정 파일과 런처**를 사용하는 방식이 일반적입니다. Hugging Face Trainer에도 `deepspeed` 인자를 통해 DeepSpeed를 통합할 수 있으며, **🤗 Accelerate 툴을 쓰면 대화형으로 설정 파일을 생성**할 수  ([DeepSpeed](https://huggingface.co/docs/peft/main/en/accelerate/deepspeed#:~:text=Configuration)) ([DeepSpeed](https://huggingface.co/docs/peft/main/en/accelerate/deepspeed#:~:text=accelerate%20config%20))164】. 예를 들어, 아래와 같은 DeepSpeed 설정(`ds_config.json`)을 준비하여 Trainer에 전달하면 ZeRO 기반 훈련이 이루어집니다:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu"
    }
  },
  "fp16": {
    "enabled": true
  }
}
```

위 설정은 ZeRO-3 단계에서 **모델 파라미터를 CPU로 오프로드**하도록 지정한 예시입니다. `TrainingArguments(..., deepspeed="ds_config.json")` 처럼 설정하면 Hugging Face Trainer가 내부적으로 DeepSpeed 엔진을 초기화하여 학습을 진행합니다. 또는 **`deepspeed.init` API를 직접 사용**해 모델, 옵티마이저를 감싼 뒤 `deepspeed.run`으로 훈련 loop을 구현할 수도 있습니다. 어떤 방법이든, **기존 PyTorch 코드를 크게 변경하지 않으면서** DeepSpeed의 이점을 얻을 수 있다는 것이 장점입니다.

DeepSpeed와 Hugging Face PEFT를 **조합하여 사용**하는 것도 가능합니다. 예를 들어 **LoRA 적용 모델을 DeepSpeed ZeRO-3로 분산 학습**하거나, **QLoRA(4비트 + LoRA)**를 DeepSpeed와 함께 활용하여 다중 GPU에서 초거대 모델을 학습하는 실험들이 보고되 ([DeepSpeed](https://huggingface.co/docs/peft/main/en/accelerate/deepspeed#:~:text=For%20DeepSpeed%20Stage%203%20%2B,models%20on%20multiple%20GPUs%20below)) ([DeepSpeed](https://huggingface.co/docs/peft/main/en/accelerate/deepspeed#:~:text=This%20section%20of%20guide%20will,by%20changing%20the%20accelerate%20config))142】. Hugging Face 가이드에서는 **8x H100 (80GB) GPU로 LLaMA-70B 모델을 LoRA+ZeRO-3 설정으로 SFT(지도파인튜닝)하는 예시**를 제공하고  ([DeepSpeed](https://huggingface.co/docs/peft/main/en/accelerate/deepspeed#:~:text=This%20section%20of%20guide%20will,by%20changing%20the%20accelerate%20config)) ([DeepSpeed](https://huggingface.co/docs/peft/main/en/accelerate/deepspeed#:~:text=accelerate%20config%20))164】. 이처럼 DeepSpeed는 Hugging Face 생태계와도 잘 맞물려 동작하며, 파인튜닝 **속도 및 확장성**을 높이는 역할을 합니다.

정리하면, DeepSpeed의 특징과 장점은 다음과 같습니다:

- **ZeRO 알고리즘**을 통한 **메모리 최적화 및 모델 분산**: 동일 자원으로 더 큰 모델 학 ([DeepSpeed](https://huggingface.co/docs/peft/main/en/accelerate/deepspeed#:~:text=DeepSpeed%20is%20a%20library%20designed,leveraging%20CPU%20resources%20during%20optimization))110】 
- **CPU/NVMe 오프로드**로 단일 GPU 메모리 한계 ([ZeRO-Offload - DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/#:~:text=ZeRO,No%20code%20changes%20are%20needed))109】 
- **고도화된 분산 병렬 학습** 지원: 수십~수백 GPU까지 효율적 스케일 아웃
- **성능 최적화 커널** 제공: DeepSpeed의 CPU Adam 옵티마이저는 기본 PyTorch 대비 5~7 ([ZeRO-Offload - DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/#:~:text=For%20large%20model%20training%2C%20optimizers,please%20see%20our%20blog%20post))119】 등

단점으로는 **환경 설정의 복잡성**이 있습니다. 설정 파일 작성, 런처 명령 등 처음 사용시 진입장벽이 있으며, 작은 규모 실험에는 과한 측면이 있을 수 있습니다. 또한 **동일한 연산이라도 약간의 오버헤드**(통신 대기 등)가 존재하므로, 모델이 충분히 크거나 분산이 필요한 경우에 가장 큰 효과를 봅니다. 그럼에도, **실무에서 수십억~수천억 파라미터** 모델을 다뤄야 한다면 DeepSpeed는 사실상 **표준 도구**로 자리잡았습니다.

## Unsloth를 활용한 고속 LLM 파인튜닝
**Unsloth**는 2023년 커뮤니티에서 등장한 **경량화 LLM 파인튜닝 라이브러리**로, **“Hugging Face 호환”**을 표방하면서도 **학습 속도를 2배 이상 높이고 메모리 사용을 40~70% 줄이는** 혁신을 보여주고  ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=1%20A100%2040GB%20Dataset%20Hugging,11.6)) ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=Unsloth%20was%20benchmarked%20across%2059,are%20in%20Unsloth%E2%80%99s%20benchmarking%20details))L92】. Unsloth의 접근법은 기존 Hugging Face `Transformers` 모델의 일부 연산을 **Triton** 기반의 맞춤 커널로 대체하여 **PyTorch 수준에서의 비효율을 제거**하는  ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=Unsloth%20works%20by%20overwriting%20some,made%20in%20the%20optimized%20code))L72】. 구체적으로, **Self-Attention, FFN 등 Transformer 핵심 모듈의 backward 연산을 수식으로 직접 유도**하여 Triton으로 구현함으로써, 같은 작업을 하면서도 **메모리 복사나 중간 연산 overhead를 최소화** ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=Unsloth%20works%20by%20overwriting%20some,made%20in%20the%20optimized%20code))L72】. 이런 **수동 최적화(manual backprop)** 기법 덕분에, **동일한 QLoRA 파인튜닝이라도 Unsloth 사용 시 학습 속도가 약 2배로 향상되고 GPU VRAM 사용은 절반 이하로 감소**하는 결과를 얻 ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=1%20A100%2040GB%20Dataset%20Hugging,11.6)) ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=Llama,18.6))L88】. 놀랍게도 **모델의 최종 성능 저하가 0%**임이 검증되었는데, 이는 Unsloth의 커널 최적화가 근본적으로 **동일한 계산을 더 효율적으로 구현**한 것이므로 정확도가 보존되기 때 ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=Unsloth%20works%20by%20overwriting%20some,made%20in%20the%20optimized%20code))L68】.

Unsloth는 **Hugging Face와의 호환성**을 강조합니다. 사용법도 매우 비슷하여, `FastLanguageModel.from_pretrained()` 함수로 모델을 불러오면 내부적으로 `transformers` 모델을 래핑한 Unsloth 모델 객체와 토크나이저를 반 ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=,returns%20the%20model%20tokenizer%20for)) ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=model%2C%20tokenizer%20%3D%20FastLanguageModel,RoPE%20Scaling%20internally%2C%20so%20choose))L22】. 예를 들어 다음과 같이 LLaMA 계열 모델을 Unsloth로 로드할 수 있습니다:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-bnb-4bit",  # HF 허브 모델명 (4-bit 양자화된 Mistral 7B)
    max_seq_length=2048                       # 최대 시퀀스 길이 (RoPE 스케일링 자동적용)
)
```

불러온 모델은 Hugging Face `transformers`와 거의 동일한 인터페이스를 제공하므로, `transformers.Trainer`나 🤗 TRL의 `SFTTrainer` 등에 그대로 넣어서 사용할 수 ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=Unsloth%20is%20a%20lightweight%20library,and%20Mistral%20architectures)) ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=most%20NVIDIA%20GPUs%20%E2%80%93from%20GTX,and%20Mistral%20architectures))-L66】. Unsloth는 현재 **LLaMA 계열(Llama-2, CodeLlama 등)과 Mistral, Qwen 등 GPT 유사 아키텍처**를 지원하며, 다양한 NVIDIA GPU(예: GTX 16GB급부터 A100/H100까지)에서  ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=Unsloth%20is%20a%20lightweight%20library,and%20Mistral%20architectures))-L63】. 특히 **FP16, BF16 혼합정밀도**도 옵션으로 켤 수 있고, **양자화된 모델(`bnb-4bit`)도 직접 로드**할 수 있어 (bitsandbytes 라이브러리 필요), Hugging Face에서 하던 4-bit QLoRA 파인튜닝을 거의 그대로 진행하면서 성능 향상을 누릴 수 있습니다.

Unsloth의 특기할 만한 기능 중 하나는 **RoPE Scaling**을 자동 처리하는 ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=,returns%20the%20model%20tokenizer%20for))-L15】. RoPE(Rotary Positional Embedding)는 GPT 계열에서 쓰이는 위치인코딩 기법인데, Unsloth 모델 로드시 `max_seq_length`를 크게 지정하면 **학습 시 더 긴 문맥길이**를 사용할 수 있도록 내부적으로 주파수를 스케일링해 줍니다. 이를 통해 본래 2048 토큰까지였던 LLaMA-2 모델도 **최대 4배 이상 긴 컨텍스트까지** 파인튜닝할 ([Fine-tuning Guide | Unsloth Documentation](https://docs.unsloth.ai/get-started/fine-tuning-guide#:~:text=,tuning))L163】, 일부 최신 모델(Llama-3.3 70B 등)은 Unsloth로 **8만~3십만 토큰 이상의 문맥 학습**도 시도되고 ([GitHub - unslothai/unsloth: Finetune Llama 3.3, DeepSeek-R1 & Reasoning LLMs 2x faster with 70% less memory! ](https://github.com/unslothai/unsloth#:~:text=with%20Llama%20%26%20Qwen%20distillations,13x%20longer))L308】. 긴 문맥 대응은 **Decoder-Only 모델의 실제 활용도**를 높이는 중요한 최적화인데, Unsloth가 이를 편리하게 지원하는 점은 실용적 장점이라 할 수 있습니다.

요약하면 Unsloth의 특징과 장점:

- **Triton 커널 기반 최적화**로 **학습속도 ~2배 향상**, **메모리 ~50% ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=1%20A100%2040GB%20Dataset%20Hugging,11.6)) ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=Llama,18.6))-L88】 (동일 하드웨어/모델 대비)
- Hugging Face **Transformers/PEFT와 완전 ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=Unsloth%20is%20a%20lightweight%20library,and%20Mistral%20architectures))-L63】 – 친숙한 API로 사용 가능
- **QLoRA(4-bit + LoRA) 지원** – 저자들이 제공한 다이나믹 4비트 양자화로 QLoRA의 미세 성능 저하 ([Fine-tuning Guide | Unsloth Documentation](https://docs.unsloth.ai/get-started/fine-tuning-guide#:~:text=improves%20accuracy%20%281%E2%80%932)) ([Fine-tuning Guide | Unsloth Documentation](https://docs.unsloth.ai/get-started/fine-tuning-guide#:~:text=We%20recommend%20starting%20with%20QLoRA%2C,LoRA%20is%20now%20largely%20recovered))L174】
- **RoPE 등 Decoder용 추가 기능** – 문맥길이 확장 등 디코더 Transformer에 유용한 최적화 제공
- **오픈소스 개발 활성화** – 콜랩 노트북, 벤치마크 스크립트 공개 등으로 재현성과 접 ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=Unsloth%20was%20benchmarked%20across%2059,are%20in%20Unsloth%E2%80%99s%20benchmarking%20details))-L92】

Unsloth의 현재 한계로는 **지원 아키텍처가 제한적**이라는 점이 있습니다. 주로 Meta의 Llama 계열과 그 파생모델에 집중되어 있고, Transformer 구조가 다른 T5(Encoder-Decoder)나 GLM 양방향 모델 등은 지원하지 않습니다. 또한 분산 학습(멀티 GPU)에 대한 언급이 적은데, 주로 단일 GPU에서의 극한 최적화에 초점이 맞춰져 있습니다. 따라서 아주 큰 모델을 여러 GPU에 나누어 학습하는 용도는 DeepSpeed만큼 주안점은 아닐 수 있습니다. 그럼에도 **단일/소수 GPU로 LLM을 최대한 빠르게 튜닝**해야 하는 실무 상황에서 Unsloth는 대단히 매력적인 선택지입니다. 예컨대, 1장의 A100으로 하루 걸리던 파인튜닝 작업을 Unsloth로 반나절에 ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=1%20A100%2040GB%20Dataset%20Hugging,11.6))-L77】, 같은 GPU에서 더 큰 배치 사이즈나 더 긴 문맥을 실험할 여유를 얻을 수 있습니다. 이것은 곧 **개발 생산성과 실험 범위의 확대**로 이어지므로, 앞으로 Unsloth와 같은 최적화 툴의 활용도는 더욱 높아질 전망입니다.

## 성능 비교 및 평가 방법
LLM 파인튜닝 기법들을 평가할 때에는 **모델의 최종 성능** 뿐 아니라 **학습 효율 지표**들도 중요합니다. 주요 비교 기준은 **메모리 사용량(VRAM)**, **학습 속도(throughput)**, **학습 안정성 및 효율성** 등이 있습니다. 아래 표는 Hugging Face 기본 방법, DeepSpeed, Unsloth의 주요 특징과 성능 상의 장단점을 정리한 것입니다:

| 접근법                     | 주요 특징 및 최적화               | 메모리 사용량              | 학습 속도                 | 비고 (장단점 요약)                  |
|----------------------------|----------------------------------|---------------------------|---------------------------|--------------------------------------|
| **Hugging Face 기본**      | - Pretrained 모델/데이터 에코시스템<br>- Trainer/Accelerate 통한 손쉬운 구현<br>- PEFT: LoRA, P-Tuning 등 지원<br>- 8/4-bit 양자화 로드 지원 | 기준 (100%)                | 기준 (1×)                | 쉬운 구현과 커뮤니티 지원이 강점. 대형 모델은 추가 최적화 필요 (예: DeepSpeed 통합 가능). |
| **DeepSpeed (ZeRO)**       | - ZeRO-1/2/3 옵티마 ([DeepSpeed](https://huggingface.co/docs/peft/main/en/accelerate/deepspeed#:~:text=DeepSpeed%20is%20a%20library%20designed,leveraging%20CPU%20resources%20during%20optimization))L110】<br>- CPU/NVMe Offlo ([ZeRO-Offload - DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/#:~:text=ZeRO,No%20code%20changes%20are%20needed))L109】<br>- 병렬화 최적 튜닝 (일괄 통신, One-bit Adam 등)<br>- 분산 훈련에 특화 | **매우 적음** (파라미터/그래디언트 분산으로 GPU당 부 ([DeepSpeed](https://huggingface.co/docs/peft/main/en/accelerate/deepspeed#:~:text=DeepSpeed%20is%20a%20library%20designed,leveraging%20CPU%20resources%20during%20optimization))L110】 | **높음** (멀티 GPU로 선형 스케일링, 단일 GPU에선 다소 오버헤드) | 초대형 모델 학습 가능 (수십억~수천억↑  ([ZeRO-Offload - DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/#:~:text=ZeRO,No%20code%20changes%20are%20needed))L109】. 초기 설정 복잡하지만, 대규모 실험엔 필수 도구. |
| **Unsloth (QLoRA 기반)**   | - Triton 커널로 모델 연 ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=Unsloth%20works%20by%20overwriting%20some,made%20in%20the%20optimized%20code))-L72】<br>- 수동 backprop으로 메모리 절약<br>- RoPE 스케일링으로 문맥 확장<br>- HF Transformers와 호환 API | **적음** (동일 QLoRA 대비 VRAM 최대 ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=Unsloth%20was%20benchmarked%20across%2059,are%20in%20Unsloth%E2%80%99s%20benchmarking%20details))-L92】) | **매우 높음** (동일 QLoRA 대비 ~ ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=1%20A100%2040GB%20Dataset%20Hugging,11.6))-L77】) | 단일/소수 GPU 환경에 최적화. 정확도 손실 없이  ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=Unsloth%20works%20by%20overwriting%20some,made%20in%20the%20optimized%20code))-L68】. 지원 모델 한정적이나 빠르게 확대 중. |

*표: Hugging Face vs. DeepSpeed vs. Unsloth의 특징 및 효율 비교*

위 비교에서 보듯이, **Hugging Face + 기본 PyTorch**는 구현 편의성 측면에서 뛰어나나 **대형 모델 학습 시 메모리 병목**이 있을 수 있습니다. DeepSpeed는 이를 해소하여 **모델 사이즈 한계를 크게 높여주지만**, 구성 복잡성과 **통신 오버헤드**가 약간 존재합니다. Unsloth는 **낮은 수준의 커스터마이징을 통해** 가장 많이 쓰이는 시나리오(예: LLaMA 계열의 SFT)에서 **최대의 속도/메모리 효율**을 끌어올린 사례입니다. 특히 QLoRA처럼 **4-bit 양자화로 인한 16-bit 대비 약간의 속도 저하**가 원래 ([Finetuning LLMs with LoRA and QLoRA: Insights from Hundreds of ...](https://lightning.ai/pages/community/lora-insights/#:~:text=Code%20Framework,which%20is%20to%20be)) ([[2305.14314] QLoRA: Efficient Finetuning of Quantized LLMs - ar5iv](https://ar5iv.labs.arxiv.org/html/2305.14314#:~:text=a%20significant%20shift%20in%20accessibility,finetunable%20on%20a%20single%20GPU))-L57】, Unsloth 최적화로 이러한 **양자화 오버헤드까지 상쇄**한 것이 큰 장점입니다.

**평가 방법**으로, **메모리 사용량**은 일반적으로 **훈련 중 최대 GPU VRAM 점유**를 측정합니다 (예: `nvidia-smi` 모니터링). DeepSpeed의 경우 ZeRO-3를 쓰면 각 GPU가 모델 일부만 갖고 있으므로 개별 GPU 메모리 사용량이 크게 줄고, 나머지는 CPU/NVMe 사용량으로  ([DeepSpeed](https://huggingface.co/docs/peft/main/en/accelerate/deepspeed#:~:text=GPU%20Peak%20Memory%20consumed%20during,begin%29%3A%200)) ([DeepSpeed](https://huggingface.co/docs/peft/main/en/accelerate/deepspeed#:~:text=GPU%20Peak%20Memory%20consumed%20during,begin%29%3A%200))-L20】. **처리 속도**는 보통 **초당 처리 토큰 수 (tokens per second)** 또는 **스텝당 시간**으로 산출합니다. 같은 하드웨어에서 배치당 토큰 throughput을 비교하면 최적화 효과를 정량화할 수 있습니다. 예컨대, Unsloth 팀은 다양한 모델/데이터셋에 대해 **초당 토큰 처리량**을 측정하여 Hugging Face 대비 **1.5×~2.7× 속도 향상**을 보고 ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=1%20A100%2040GB%20Dataset%20Hugging,11.6)) ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=Free%20Colab%20T4%20Dataset%20Hugging,18.6))-L87】.

학습 효율 이외에, **모델 성능 평가** 또한 필수입니다. 모델이 지도파인튜닝을 통해 목표 작업에 얼마나 향상되었는지, 또는 혹시 **기존 지식을 훼손**하지 않았는지 등을 확인해야 합니다. **Decoder-Only LLM**의 경우 일반적으로 **텍스트 생성 품질**이나 **다양한 다운스트림 태스크 성능**으로 평가합니다. 예를 들어, **지도학습으로 대화형 모델**을 튜닝했다면 **ChatGPT와 유사한 벤치마크(Vicuna Benchmark 등)**에서 대화 품질을 측정하거나, Human 평가 혹은 GPT-4를 활용한 비교 평가를 수행할 수 ([[2305.14314] QLoRA: Efficient Finetuning of Quantized LLMs](https://ar5iv.org/abs/2305.14314#:~:text=full%2016,innovations%20to%20save%20memory%20without)) ([](https://openreview.net/pdf?id=OUIFPHEgJU#:~:text=on%20the%20Vicuna%20,Table%204))L107】. QLoRA 논문에서는 **GPT-4 기반 자동 평가**를 통해, 65B 모델을 QLoRA로 미세조정한 Guanaco가 ChatGPT 대비 99.3% 수준에 도달했음을  ([[2305.14314] QLoRA: Efficient Finetuning of Quantized LLMs](https://ar5iv.org/abs/2305.14314#:~:text=full%2016,innovations%20to%20save%20memory%20without))-L18】. 이처럼 **모델 출력의 정량·정성 평가**를 통해 파인튜닝의 효과를 검증해야 합니다. 또한 **perplexity**(언어모델의 로그확률 지표)도 사용되는데, 원래 모델 대비 퍼플렉서티 변화로 **과적합 여부나 일반화 성능**을 가늠할 수 있습니다. 최신 연구에 따르면 **파인튜닝 데이터의 품질이 데이터량보다 중요**하며, 고품질 소량 데이터로도 강력한 성능을 낼 수 ([](https://openreview.net/pdf?id=OUIFPHEgJU#:~:text=Guanaco%2C%20,strong%20Vicuna%20chatbot%20benchmark%20performance)) ([](https://openreview.net/pdf?id=OUIFPHEgJU#:~:text=analyze%20trends%20in%20the%20trained,strong%20Vicuna%20chatbot%20benchmark%20performance))L142】. Meta의 **LIMA 연구(2023)**에서는 LLaMA 65B 모델을 **엄선된 1000개의 예시**만으로 지도학습 파인튜닝 하였을 때 GPT-4 등 거대 모델에 필적하는 성능을 달성하기도 ([Paper page - LIMA: Less Is More for Alignment - Hugging Face](https://huggingface.co/papers/2305.11206#:~:text=Face%20huggingface,learning%20or%20human%20preference%20modeling))-L18】. 이는 **사전학습된 거대 LM의 잠재력을 끌어내는 데 있어, 방대한 양의 미세조정 데이터보다 인간 전문가가 고른 핵심 데이터가 효과적**일 수 있음을 시사합니다.

마지막으로, **Decoder-Only Transformer 최적화 기법**들을 정리하면 다음과 같습니다:

- **양자화(Quantization)**: 16-bit 대신 8-bit, 4-bit로 모델 가중치를 표현해 메모리 감소 (예: QLoRA의 4-bit NF ([Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes#:~:text=In%20few%20words%2C%20QLoRA%20reduces,on%20a%20single%2046GB%20GPU))L205】. 적절한 양자화는 **성능 유지하면서 메모리 4배 절약** 가능.
- **파라미터 효율 기법(PEFT)**: ([Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes#:~:text=More%20specifically%2C%20QLoRA%20uses%204,in%20the%20original%20LoRA%20paper))L210】, Adaptor, Prefix-Tuning 등으로 **소수의 파라미터만 학습**하여 연산/메모리 효율 개선.
- **Flash Attention 등 메모리 효율 Attention**: 시퀀스 길이가 길어질 때 메모리 사용을 줄이고 속도를 높이는 **최적화 Attention 알고리즘**. PyTorch 2.x에서는 이러한 **SDPA(Scaled Dot-Product Attention)**가 기본 통합되어 있어 성능 향 ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=Unsloth%20was%20benchmarked%20across%2059,are%20in%20Unsloth%E2%80%99s%20benchmarking%20details))-L92】.
- **Gradient Checkpointing**: 중간 활성값을 저장하지 않고 재계산하는 기법으로, **GPU 메모리 사용을 큰 폭으로 절감** (대신 계산량 증가). 대형 모델 파인튜닝에 거의 필수적으로 쓰입니다.
- **Mixed Precision Training**: FP32 대신 **FP16/BF16** 등을 사용하여 **연산 속도와 메모리 사용 최적화**. 최근 GPU는 BF16/FP16 성능이 뛰어나므로, 정확도에 큰 문제없이 활용.
- **분산 병렬화**: 모델 병렬화(레이어를 여러 GPU에 분할), 데이터 병렬화, 파이프라인 병렬화 등 조합으로 **하드웨어 자원 활용 극대화**. DeepSpeed, FSDP, Megatron-LM 등이 지원.
- **동적 장비 메모리 활용**: GPU와 CPU, 디스크를 모두 활용하여 **계산 자원 대비 최대 메모리 활용** (ZeRO-Offload/Infinit ([ZeRO-Offload - DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/#:~:text=ZeRO,No%20code%20changes%20are%20needed))L109】.
- **최신 옵티마이저 사용**: AdamW 외에 LAMB, Lion 등의 대안 옵티마이저나, DeepSpeed의 One-bit Adam처럼 **통신량을 줄인 분산 옵티마이저**로 효율 개선.
- **정규화 및 안정화 기법**: 대규모 LM 파인튜닝 시 **러닝레이트 워밍업**, **학습률 스케줄**, **Gradient Clipping** 등으로 안정적 수렴을 도모. 이는 간접적으로 효율(재시도 감소 등)에 기여.
- **Continuous Pretraining과 SFT 결합**: 경우에 따라 **사전학습 연장(Continued Pretraining)** 후 SFT를 하면 더 좋은 결과를 얻거나, SFT 도중 **기존 지식 유지**를 위한 **混合 사전학습 데이터 사용** 등의 기법도 연구되고 ([Fine-tuning Guide | Unsloth Documentation](https://docs.unsloth.ai/get-started/fine-tuning-guide#:~:text=the%20accuracy%20loss%20for%20QLoRA,LoRA%20is%20now%20largely%20recovered))L174】.

## 최신 연구 동향 및 결론
최근 2년간 LLM 파인튜닝 분야는 **“더 적은 자원으로 더 큰 모델을 효과적으로 다루는 법”**에 집중되어 왔습니다. **QL ([[2305.14314] QLoRA: Efficient Finetuning of Quantized LLMs](https://ar5iv.org/abs/2305.14314#:~:text=We%20present%20QLoRA%2C%20an%20efficient,innovations%20to%20save%20memory%20without))-L18】의 등장으로 촉발된 **저비트 양자화 + 어댑터 학습** 패러다임은 현재 업계 표준으로 자리잡았고, 이를 넘어 **아직 실험 단계인 3비트, 2비트** 미세튜닝 연구도 진행중입니다. 또한 **LORA의 변형**으로서 중요도가 높은 레이어에 가중치를 더 할당하는 **AdaLoRA** 등의 기법도 제안되었습니다. 한편, **파인튜닝 데이터 확보** 측면에서는 Stanford의 **Alpaca** 프로젝트처럼 **기존 모델(예: GPT-3)를 이용한 Self-Instruct 데이터 생성**이 유행하여, 비교적 저렴하게 지도학습 데이터를 모으는 흐름이 있습니다. 이를 통해 탄생한 **Vicuna**, **WizardLM**, **OpenAssistant** 등의 **오픈소스 대화형 모델**들은 모두 공개 데이터나 생성 데이터로 SFT된 사례들입니다. 성능 면에서, 앞서 언급한 **LIMA (Less is More for Alignme ([Paper page - LIMA: Less Is More for Alignment - Hugging Face](https://huggingface.co/papers/2305.11206#:~:text=Face%20huggingface,learning%20or%20human%20preference%20modeling))-L18】 연구는 **고품질 소규모 데이터의 위력**을 보여주었고, OpenAI도 **InstructGPT 논문(2022)**에서 인간 피드백 외에 **초기 단계의 슈퍼바이즈드 파인튜닝(SFT)**이 핵심적으로 중요함을 밝힌 바 있습니다. 최근에는 **RLHF**(강화학습 휴먼 피드백) 대신 **DPO**(Direct Preference Optimization)나 **RLAIF**(AI 피드백) 등 **순수 지도 신호만으로 선호도를 학습**하려는 시도도 나오고 있어, **지도 파인튜닝의 범위가 확장**되고 있습니다.

정리하면, **Decoder-Only LLM의 지도 파인튜닝**은 여전히 **모델 성능 개선과 효율적 학습**을 양립하기 위한 다양한 연구로 활발히 진화하고 있습니다. Hugging Face, DeepSpeed, Unsloth와 같은 도구들은 이러한 연구 성과를 현업에 적용하는 다리 역할을 하며, 각기 **사용자 요구와 환경에 맞는 솔루션**을 제공합니다. 실무에서는 세 가지 접근법을 **상황에 따라 조합**하기도 합니다. 예를 들어, **중간 규모 모델은 Unsloth로 싱글 GPU 빠르게 튜닝**하고, **초거대 모델은 DeepSpeed로 멀티 GPU 분산 학습**하며, 전반적인 워크플로우는 Hugging Face 에코시스템으로 관리하는 식입니다. 중요한 것은 **모델의 목표와 제약에 맞춰 최적의 기법을 선택**하는 것입니다. 앞으로도 하드웨어와 알고리즘 측면의 발전으로 LLM 파인튜닝은 더욱 최적화되고 대중화될 것이며, **“더 낮은 비용으로 더 똑똑한 모델”**을 만드는 방향으로 나아갈 것입니다.

**참고 문헌 및 링크:** 최신 파인튜닝 기법과 사례에 대한 자세한 내용은 Hugging Face 블로그 및 각 논문의 원문을 참고하시기 바랍니다. 아래는 본 문서에서 언급된 자료들의 출처입니다.

- Hugging Face 블로그: *Making LLMs even more accessible with 4-bit quantization and Q ([Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes#:~:text=In%20few%20words%2C%20QLoRA%20reduces,on%20a%20single%2046GB%20GPU)) ([Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes#:~:text=QLoRA%20tuning%20is%20shown%20to,the%20power%20of%20QLoRA%20tuning))L222】
- Hugging Face 블로그: *Make LLM fine-tuning 2x faster with Unsloth and ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=Unsloth%20works%20by%20overwriting%20some,made%20in%20the%20optimized%20code)) ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=1%20A100%2040GB%20Dataset%20Hugging,11.6))-L80】
- Hugging Face Docs: *DeepSpeed & Accelerate Integration G ([DeepSpeed](https://huggingface.co/docs/peft/main/en/accelerate/deepspeed#:~:text=DeepSpeed%20is%20a%20library%20designed,leveraging%20CPU%20resources%20during%20optimization)) ([ZeRO-Offload - DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/#:~:text=ZeRO,No%20code%20changes%20are%20needed))L109】
- Unsloth 공식 문 ([Fine-tuning Guide | Unsloth Documentation](https://docs.unsloth.ai/get-started/fine-tuning-guide#:~:text=,tuning)) ([Make LLM Fine-tuning 2x faster with Unsloth and  TRL](https://huggingface.co/blog/unsloth-trl#:~:text=1%20A100%2040GB%20Dataset%20Hugging,11.6))-L77】
- QLoRA 논문 (Dettmers et al.,  ([[2305.14314] QLoRA: Efficient Finetuning of Quantized LLMs](https://ar5iv.org/abs/2305.14314#:~:text=We%20present%20QLoRA%2C%20an%20efficient,innovations%20to%20save%20memory%20without)) ([[2305.14314] QLoRA: Efficient Finetuning of Quantized LLMs](https://ar5iv.org/abs/2305.14314#:~:text=sacrificing%20performance%3A%20%28a%29%204,of))-L27】
- LIMA 논문 (Zhou et al.,  ([Paper page - LIMA: Less Is More for Alignment - Hugging Face](https://huggingface.co/papers/2305.11206#:~:text=Face%20huggingface,learning%20or%20human%20preference%20modeling))-L18】

