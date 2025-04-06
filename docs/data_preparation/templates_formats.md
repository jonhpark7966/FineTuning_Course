# 템플릿과 포맷

## 데이터 템플릿 설계
- 효과적인 템플릿 구조
- 역할 기반 포맷 (사용자/어시스턴트)
- 시스템 지시 활용

## jinja 템플릿 활용
- jinja2 기본 문법
- 파인튜닝 데이터에 jinja 적용하기
- 조건부 및 반복 구문 활용

## 채팅 형식 vs 명령어 형식
- 채팅 기반 학습 데이터 구성
- 명령어-응답 형태 데이터 구성
- 각 형식의 적합한 사용 상황 


# chat_template 이해와 활용

<div style="text-align: center;">
  <img src="../../rscs/chat_template.png" alt="chat_template_illustration">
</div>

> chat_template은 LLM 파인튜닝과 추론에서 대화 형식을 일관되게 구성하는 핵심 요소입니다. 모델마다 다른 대화 포맷을 표준화하여 성능을 최적화합니다.

## chat_template의 정의와 역할

chat_template은 대화형 LLM에서 **여러 발화들을 하나의 프롬프트로 구성하는 템플릿**입니다. 시스템 메시지, 사용자 질문, 모델 답변 등 **각 발화의 역할(role)을 표시**하고, 이를 **하나의 연속된 토큰 시퀀스**로 합치는 형식을 말합니다.

### 기본 개념

- **역할 구분**: 대화에서 누가 말하는지(system, user, assistant) 명확히 표시
- **특수 토큰**: 각 발화의 시작과 끝을 알리는 제어 토큰 사용
- **일관된 포맷**: 모델이 학습한 형식과 동일한 입력 구조 유지

LLM은 본질적으로 연속된 텍스트로 학습되므로, 대화 맥락을 주기 위해서는 각 발화 앞뒤에 **특수 토큰이나 구분자를 넣어 역할 정보를 주입**해야 합니다. 이러한 템플릿이 중요한 이유는, **모델이 학습된 포맷과 일치하는 입력을 제공해야** 모델이 맥락을 정확히 이해하고 올바른 답변을 내기 때문입니다.

```
모델이 기대하는 형식과 다른 프롬프트를 제공하면 → 성능 저하
올바른 chat_template으로 포맷된 프롬프트 제공 → 최적 성능
```

## 주요 LLM 프레임워크별 chat_template 비교

각 프레임워크나 API는 대화 형식을 표현하는 방식이 다릅니다. 주요 모델별 chat_template 구성을 비교해보겠습니다.

### Hugging Face Transformers

Hugging Face는 `transformers` 라이브러리에서 **채팅 템플릿을 토크나이저에 내장**하는 방식을 제공합니다. `AutoTokenizer` 객체에 `chat_template` 속성이 있으며, 내부적으로 **Jinja2 템플릿 문자열**로 정의됩니다.

```python
# 토크나이저에서 chat_template 확인
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
print(tokenizer.chat_template)

# 대화 메시지 리스트
messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "Tell me about chat templates"}
]

# 템플릿 적용
formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False)
print(formatted_chat)
```

Hugging Face는 기본적으로 **ChatML 포맷**을 채택하여, 별도 지시 없으면 user/system/assistant 역할에 대해 ChatML 형식을 사용합니다. 반면 특정 모델들은 자체 템플릿을 가질 수도 있습니다.

### OpenAI Chat Completion API

OpenAI의 GPT-3.5/GPT-4 챗 completions API에서는 사용자가 별도의 템플릿 문자열을 작성할 필요는 없습니다. 대신 **`messages` 리스트**에 각 대화 turn을 **역할과 내용 필드로 제공**하면 OpenAI가 내부적으로 적절한 포맷(일명 *ChatML*)으로 처리합니다.

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}
  ]
}
```

OpenAI는 이 형식을 모델 학습 및 추론에 일관되게 사용하며, 실제 모델 내부적으로는 `<|im_start|>user`, `<|im_start|>assistant` 등의 토큰으로 구성된 **Chat Markup Language (ChatML)**로 변환되어 처리됩니다.

### Mistral 및 LLaMA 계열

Mistral-7B-Instruct와 같은 LLaMA 기반 공개 모델들은 **Meta의 LLaMA-2 Chat** 포맷과 유사한 템플릿을 사용합니다. 대표적으로 **Mistral-7B Instruct** 모델의 공식 템플릿은 다음과 같은 구조를 취합니다:

```
<s>[INST] 사용자질문 [/INST] 모델답변</s>
```

여기서 `<s>`는 시작(BOS) 토큰, `[INST]`와 `[/INST]`는 **사용자 명령 시작/종료**를 뜻하는 특수 토큰이며, 그 뒤에 모델의 응답이 나오고 `</s>` (EOS 토큰)로 대화를 닫습니다.

Mistral의 Jinja 템플릿 예시:

```jinja
{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}
```

이러한 LLaMA 스타일 템플릿에서는 별도의 "system" 역할 토큰이 없는 대신, **필요한 시스템 지침을 첫 번째 [INST] 블록 내부에 포함**시키는 방식으로 활용합니다.

### 기타 모델 (Anthropic Claude 등)

다른 프레임워크나 모델들도 각기 다른 채팅 포맷을 갖습니다. 예를 들어 **Anthropic의 Claude** 모델은 학습 시 대화를 **"Human:"과 "Assistant:"**라는 텍스트 태그로 구분하는 형식을 사용합니다.

```
Human: 안녕하세요, 오늘 날씨가 어때요?
