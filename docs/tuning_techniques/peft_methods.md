# Parameter-Efficient Fine-Tuning (PEFT)


Parameter-Efficient Fine-Tuning (PEFT)는 LLM의 모든 파라미터를 업데이트하지 않고도 모델을 튜닝하는 방법을 칭합니다.  
실전에서는 LLM 을 Full Tuning 하기가 너무 비싸요...... 

## 필요성

대형 언어 모델의 전체 파라미터를 미세 조정(full fine-tuning)하는 것은 막대한 컴퓨팅 자원을 필요로 합니다.

2025년 3월 기준, 연구용으로 가장 많이 사용되는 32B 모델을 Full Tuning 하려면 대략 300GB 의 GPU 메모리가 필요합니다. (여러 테크닉을 통해 줄이는 것이 당연히 가능하긴 합니다)

300GB VRAM 확보하려면, H100 (80GB) 4장... 구입하려면 2~3억은 할 것 같네요. 저도 그렇고 많이들 대여해서 사용하는데요 그래도 비쌉니다.   

| 방법 | 정밀도 | 7B | 13B | 30B | 70B | 110B |
|------|--------|-----|------|------|------|------|
| Full | 16비트 | 67GB | 125GB | 288GB | 672GB | 1056GB |
| LoRA | 16비트 | 15GB | 28GB | 63GB | 146GB | 229GB |
| QLoRA | 8비트 | 9GB | 17GB | 38GB | 88GB | 138GB |
| QLoRA | 4비트 | 5GB | 9GB | 20GB | 46GB | 72GB |

> BF16, AdamW 기준 대략적인 계산입니다 [VRAM 요구사항 참고](https://modal.com/blog/how-much-vram-need-fine-tuning)

---

PEFT는 이러한 문제를 해결하기 위해 모델의 극히 일부 파라미터만 조정하거나 적은 양의 신규 파라미터를 추가 학습하는 접근 입니다.  

학습 비용이 절감되는 것 뿐 아니라, 여러 다운스트림 작업에 대해 기본 모델은 공유하고 작업별 어댑터만 교체하는 효율적 배포도 가능합니다. 

이렇게 비용을 줄였는데 Full Tuning 보다 성능이 떨어지지 않을까요? 하면. 음 그럴 수도 있고 아닐 수도 있습니다. 차차 보겠습니다. 




## LoRA (Low-Rank Adaptation)

LoRA는 사전훈련된 모델의 가중치를 동결한 채, low-rank decomposition를 통해 효율적으로 모델을 미세 조정하는 기법입니다, 사실상 최고 인기 기법입니다.  
LLM 이 아닌 Diffusion Model 에서도 많이 사용되고 있습니다. 

### 작동 원리

LoRA의 핵심 아이디어는 weight 업데이트를 low-rank 행렬의 곱으로 근사하는 것입니다, 사실 단순한 선형대수 문제입니다.  

- 기존 Weight Matrix
$$W \in \mathbb{R}^{d \times k}$$

- LoRA 업데이트 
$$ \Delta W = A \times B $$ 
$$A \in \mathbb{R}^{d \times r}$$
$$B \in \mathbb{R}^{r \times k}$$
$$r \ll \min(d, k)$$

- 최종 적용
$$W' = W + \alpha \cdot \Delta W$$ 
(여기서 alpha는 스케일링 하이퍼파라미터 입니다.)

이 방식은 학습 가능한 파라미터 수를 크게 줄입니다. 하이퍼 파라미터 설정하기 나름인데요, 1% 미만의 weight만 업데이트 하는 것이 일반적인 것 같아요.

<div style="text-align: center;">
  <img src="../../rscs/lora.png" alt="LoRA Architecture">
  <p><em>출처 - <a href="https://arxiv.org/abs/2106.09685">LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS</a></em></p>
</div>

### 주요 하이퍼파라미터

Unlosth의 LoRA 설정 코드를 발췌했습니다. 

```python
model = FastLanguageModel.get_peft_model(
    model,

    r=16,  # Choose any positive number! Recommended values include 8, 16, 32, 64, 128, etc.
    # Rank parameter for LoRA. The smaller this value, the fewer parameters will be modified.

    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    # Specify the modules to which LoRA will be applied

    lora_alpha=32,
    # Alpha parameter for LoRA. This value determines the strength of the applied LoRA.

    ...
)
```

- **랭크(r)**: 저랭크 분해의 차원으로, 일반적으로 4, 8, 16 등의 값 사용, 얼마나 많은 parameter를 업데이트 할지 결정 합니다, 16 이상을 사용하는 것을 권장합니다.  
- **타겟 모듈**: LoRA를 적용할 모델 내 특정 모듈 (예: attention의 query, key, value 행렬) 들을 결정합니다, 업데이트할 layer 들을 설정하는 것이죠.
- **알파(α)**: 업데이트 스케일링 팩터, 보통 r의 2배 정도로 설정을 권장합니다, LoRA 의 강도를 결정합니다.

### 장점

- 메모리 사용량을 크게 줄임, update 되는 weight는 1% 이하 수준 
- 추론 시 지연 시간 증가 없음 (LoRA 가중치를 원본과 merge 가능)
- 다양한 작업에 대해 기본 모델은 공유하고 작은 LoRA 가중치만 교체 가능
- 많은 경우에 full-tuning 과 비슷한 성능 (?!!!!) 


## LoRA VS Full Tuning

LoRA 는 파라미터의 아주 일부분만 업데이트를 하는데... 정말 괜찮을까요? 이 부분에 의문이 많이 드실 수 있습니다, 저도 그랬고요. 

| 💡 필자의 의견 & 경험담 |
|---------|
| 제 개인적인 경험에 의하면, Fine-Tuning으로 우리가 하고 싶은 일이 대단한 일이 아니기 때문에 LoRA 충분한 것 같습니다. 말투를 교정하거나 일부분의 도메인 지식을 추가하거나... 전체 LLM 이 가진 능력에 비하면 아주 미비한 수준이라고 느껴집니다. 그리고 아주 큰 차원의 행렬에서 뽑아낸 dominant 한 파라미터는 숫자의 갯수에 비해 큰 영향력을 가지기도 하죠. 어쨌든 저는 LoRA 로 파인튜닝시, 목적을 대부분 달성했습니다.  |


남들의 주장들도 살펴보겠습니다. 


<div style="text-align: center;">
  <img src="../../rscs/lora_vs_fulltuning.png" alt="LoRA vs Full Tuning">
  <p><em>LoRA vs Full Tuning</em></p>
</div>

- [LoRA Learns Less and Forgets Less (24.05)](https://arxiv.org/abs/2405.09673) 에 따르면, 덜 배우고 덜 잊는다고 합니다, 당연히 그럴 것 같습니다. 
- [Fine-Tuning LLMs: LoRA or Full-Parameter? An in-depth Analysis with Llama 2 (23.09)](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2) 에 따르면, 95% vs 97% 의 성능 차이로 실사용에서 문제가 되지 않는 수준이라고 했습니다. 
- [LoRA vs Full Fine-tuning: An Illusion of Equivalence (24.10)](https://arxiv.org/abs/2410.21228) 에 따르면, 
    -  LoRA로 업데이트된 가중치 행렬의 특이값 분포를 분석했는데, LoRA의 경우 **기존 사전학습 특성 공간에 없던 새로운 고차원 특이벡터들 ("Intruder Dimension")**이 등장함을 발견했습니다​, 반면 Full Fine-tuning 모델은 사전학습된 특성 공간을 보다 일관되게 유지했다고 합니다​
    -  rsLORA (rank-stabilized LoRA) 라는 확장 기법이 있는데, 이 경우에는 Full Tuning 과 비슷한 분포를 보인다고 하는 군요. 
    - 수학적 차이점이 이제 발견되어 가는 것 같습니다, 요약하자면 성능이 비슷하더라도 모델이 이해하는 내용은 표현이 다르다는 것이고요, 그래서 특정한 상황에서는 차이를 보일 수 있겠네요.

---

요약하겠습니다.

-  범용적인 SFT 에서는 큰 차이가 없을 것입니다화
-  복잡한 과제 (수학이나 코딩, 혹은 CPT와 같은 큰 도메인 변화) 에서는 LoRA 가 불리한 경우가 꽤 있습니다. 
-  Fine Tuning 의 부작용 중 하나는 원 모델의 지식이나 능력을 잊어먹는 것인데 (Catastrophic Forgetting), 이 문제는 LoRA 에서는 덜 발생합니다. 


### QLoRA (Quantized LoRA)

QLoRA는 LoRA의 확장으로, 4비트 양자화(quantization)와 LoRA를 결합하여 메모리 효율성을 극대화한 기법입니다.  
Quantization 에 대한 내용은 [양자화 문서](../tuning_techniques/quantization.md) 를 참조하세요. 

#### 작동 원리

1. Pre-Trained 모델 weights를 4비트 정밀도로 양자화하여 메모리 사용량을 줄입니다.
2. 양자화된 weights는 동결하고 LoRA 어댑터만 학습합니다. 
3. forward/backward 계산 시 필요한 부분만 다시 고정밀도 (ex. bf16) 으로 올려서 연산 수행합니다. 

QLoRA는 다음과 같은 혁신적 기술을 도입했습니다:

- **4비트 NormalFloat (NF4)**: 정규 분포된 가중치에 최적화된 커스텀 4비트 데이터 타입
- **Double Quantization**: 양자화 상수도 다시 양자화하여 추가 메모리 절약
- **Paged Optimizers**: 옵티마이저 상태를 CPU와 GPU 사이에 효율적으로 관리


#### 장점

- 메모리 사용량을 FP16 대비 최대 4배 이상 절감해서, 70B 규모 모델도 단일 48GB GPU에서 미세 조정 가능하다고 합니다.
- [QLoRA](https://arxiv.org/pdf/2305.14314)에 따르면 Guanaco 시리즈에서 16bit 기반 fine-tuned 모델과 성능을 똑같이 보였다고 합니다. 

성능 저하 부분에 대해서는 논란의 여지가 많은데요... 당연히 언제나 비슷한 성능을 보이지 못합니다. 위 논문의 주장은 베스트 케이스죠.  
제 경험상 QLoRA 당연히 LoRA 보다 안 좋은 경우가 많습니다. 빠르게 가능성을 테스트해보고 싶을 때 사용하는 것이 좋습니다.  


### QLoRA 적용 예시

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 4비트 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                   # 4비트 로딩 활성화
    bnb_4bit_quant_type="nf4",           # NF4 양자화 사용
    bnb_4bit_use_double_quant=True,      # 이중 양자화 활성화
    bnb_4bit_compute_dtype=torch.bfloat16  # 연산 시 BF16 사용
)

# 양자화된 모델 로드
model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA 구성 및 적용
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
qlora_model = get_peft_model(model_4bit, lora_config)
```


### 기타 PEFT 기법


- **어댑터(Adapter)**: 트랜스포머 각 층에 작은 신경망 모듈을 삽입하여 학습
- **프리픽스 튜닝(Prefix Tuning)**: 각 층에 훈련 가능한 가짜 토큰 벡터(prefix)를 추가
- **프롬프트 튜닝(Prompt Tuning)**: 입력 임베딩 공간에서 학습 가능한 소프트 프롬프트 추가
- **BitFit**: 모델의 바이어스 파라미터만 업데이트

[PEFT Methods](https://huggingface.co/blog/samuellimabraz/peft-methods) 에 또 다른 많은 방법들이 있으니 참조하시면 되겠습니다. 





## 결론

이 글을 보시는 분들이 가장 필요한 것은 "그래서 뭘 써야해?" 라는 질문에 대한 대답이겠죠.

1. 나는 GPU 가 진짜로 넉넉하다! (ex. H100x8)  -> 그냥 BF16에 풀튜닝 하세요.
2. 나는 GPU 가 애매하다.... -> QLoRA 써보시고, 잘 안되면 LoRA 써보세요. 한정된 메모리에는 Quantization 된 큰 모델을 쓰시는 걸 추천 드립니다.
3. 추론으로 적용할 타겟 하드웨어 메모리가 너무 빡빡하다 -> 어차피 큰 모델 못 쓰실테니 LoRA 써보세요.



| 💡 필자의 의견 |
|---------|
| 저는 하드웨어 자원을 큰 모델의 풀튜닝에 투자할 시간에 데이터의 품질을 올리는 것에 투자하는 것이 훨씬 더 좋은 선택이라고 생각합니다. 제 개인적 경험에 기인한 추천 가이드입니다. 
최근 딥시크가 FP8 혼합 훈련 기법을 아주 잘 공개했는데, 이 방법이 대중화가 된다면 25년에는 저도 생각이 바뀔지도 모르곘네요. 그렇다면 이 문서를 업데이트 하겠습니다. |



## 참고 문헌

1. Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.
2. Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. arXiv preprint arXiv:2305.14314.
3. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., ... & Gelly, S. (2019). Parameter-efficient transfer learning for NLP. In International Conference on Machine Learning (pp. 2790-2799). PMLR.
4. Li, X. L., & Liang, P. (2021). Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190.
5. Lester, B., Al-Rfou, R., & Constant, N. (2021). The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691.
6. Rasley, J., Rajbhandari, S., Ruwase, O., & He, Y. (2020). DeepSpeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 3505-3506).