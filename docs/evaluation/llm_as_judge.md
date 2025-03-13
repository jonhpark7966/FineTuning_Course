# LLM as Judge 와 평가 프레임워크

## LLM as Judge

LLM 이 채점을 합니다. 평가자로서 LLM 을 사용하고, 기준도 넣어주고 LLM 이 평가 점수와 이유까지 내줍니다. (실제 사람) 전문가보다 더 잘 평가한다면 강력하고, 시간도 단축되고, 비용도 줄고, 좋겠죠. 평가를 잘 하게 만드는 것 또한 하나의 LLM application 이라, 배보다 배꼽이 더 커질 수도 있습니다.  
(요즘엔 GPT4.5 를 필두로 LLM 이 비싸져서 사람이 더 싼 경우도 있는 것 같네요 ... ;;)


## 평가 예시

### reference-free 한 평가

답변이 공격성이 있는 지, 부정적인지, 간결한지, 등 정답지와 무관한 평가를 진행할 수도 있습니다.  
주관적으로 평가해야하는 것들이라 LLM 이 평가하는 것이 좋아요.  

> LangSmith 와 같은 도구에서는 conciseness, harmfulness, maliciousness 등 여러 기준에 대해 평가하는 LLM judge 를 템플릿 처럼 만들어서 제공합니다. 매우 유용합니다. 원버튼에 다양한 평가를 바로 수행가능하니까요.


LangSmith 에서 프롬프팅한 reference-free 평가 예시를 보겠습니다. 

출처 - [Answer Helpfulness](https://smith.langchain.com/hub/langchain-ai/rag-answer-helpfulness)

```
system

You are a teacher grading a quiz. 
You will be given a QUESTION and a STUDENT ANSWER. 
Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Score:
A score of 1 means that the student's answer meets all of the criteria. This is the highest (best) score. 
A score of 0 means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset.
```
```
human

STUDENT ANSWER: {{student_answer}}
QUESTION: {{question}}
```

이 템플릿에 질문과 LLM의 응답을 넣어주면, LLM 은 답변이 helpful 했는지 평가를 해줍니다.


### reference 가 필요한 평가

데이터셋의 적혀있는 예상 결과값과 비교하여 평가를 내릴 수도 있습니다.  
LLM application 에 따라 평가 로직 개발이 어려울 수도 있습니다.

> LangSmith 와 같은 도구에서는  QA correctness, Context QA, Chain of Thought QA 와 같이 얼마나 대답이 정확한지 reference 와 비교하여 결과를 내리주기도 합니다.  

위 예시와 비슷하게 LangSmith 에서 프롬프팅한 reference-related 평가 예시를 보겠습니다.

출처 - [Answer Correctness](https://smith.langchain.com/hub/langchain-ai/rag-answer-vs-reference)

```
system

You are a teacher grading a quiz. 

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.

Score:
A score of 1 means that the student's answer meets all of the criteria. This is the highest (best) score. 
A score of 0 means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset.
```
```
human
QUESTION: {{question}}
GROUND TRUTH ANSWER: {{correct_answer}}
STUDENT ANSWER: {{student_answer}}
```

이 템플릿에 질문과 나와야할 정답, 그리고 LLM의 응답을 넣어주면, LLM 은 답변이 맞았는지 평가해줍니다.



### 첨언 - LLM 만을 위한 것이 아닌 LLM 시스템을 위한 것

위 평가 예시나 답변을 보시면 아시겠지만, LLM 모델 그 자체만을 위한 것은 아닙니다. Embedding 모델을 사용해서 검색한 결과가 괜찮은지 평가를 할 수도 있고요, RAG 시스템의 전반적인 성능 평가일 수도 있고, LLM Application 의 실행 중에 중간 평가로 사용될 수도 있습니다. LLM 이랑 연관되지 않은 In/Out 을 LLM 으로 평가하는 것도 가능하죠. 학생들의 논술 채점으로도 사용할 수도 있겠네요.  

본 문서는 LLM 튜닝을 위한 문서이지만, 평가 기법 자체는 훨씬 범용성이 뛰어나다는 점을 짚고 넘어갑니다. 


## Frameworks

이를 위한 여러 프레임워크들이 있습니다. 저는 개인적으로 [LangSmith](https://www.langchain.com/langsmith)를 선호합니다. 개발 경험이 좋았고, 신기능이 거의 제일 빨리 들어오고, 평가 뿐 아니라 데이터 관리 측면에서도 좋은 면이 있어서요. 단점은 trial 을 제외하면 유료입니다... 


### LangSmith

LangSmith는 LangChain 팀이 개발한 평가 및 관측(Observability) 플랫폼으로, LLM 애플리케이션의 작동을 추적하고 품질을 모니터링하는 데 유용합니다.
참고로 LangSmith는 제가 따로 작성한 [사용법 문서](https://jonhpark7966.github.io/LangSmith_Course/#quick-start)도 있으니 필요하신 분들은 참조하세요.

- 복잡한 프롬프트 시퀀스 추적 및 품질 평가가 잘 준비되어있습니다. 
- 편향성(Bias) 탐지나 안전성 검토 같은 특화된 평가 기능 포함
- Hosted 서비스 형태로 제공, Closed source, 유료로 사용하면 설치형으로 직접 호스팅 할 수도 있습니다. (비슷한 오픈소스들도 많이 있어요. LangFuse 라던가...)
- LangChain 기반 workflow와 원활하게 연계
- LLM 상호작용과 프롬프트 체인을 로그로 기록하고 분석하는 도구 제공

LangSmith는 주로 LangChain으로 다단계 체인을 구성한 LLM 앱 개발 시 디버깅과 품질 평가에 활용됩니다.  
Fine-Tuning 의 관점에서 보면,  
모델을 튜닝해서 "모델만 배포하고 나는 이제 끝!" 하는 경우는 별로 없겠죠. 
결국 모델은 시스템 속으로 들어가서 일을 하게 될텐데요, 그 시스템 관점에서 평가를 하게 되기 떄문에 유용한 프레임워크라고 볼 수 있겠습니다. 



### OpenAI Evals

OpenAI Evals는 OpenAI가 자체 모델 평가를 위해 개발하여 오픈소스로 공개한 프레임워크입니다.  
LangSmith 처럼 시스템을 평가한다기 보다는 모델 그 자체의 평가에 더 집중한 프레임워크라고 보면 되겠습니다.  

참조 - [GitHub 저장소](https://github.com/openai/evals)


- 다양한 벤치마크 평가 세트의 레지스트리 제공
- GPT 시리즈를 비롯한 LLM들의 다양한 측면의 성능 테스트 가능
- 사용자가 자신만의 평가(Eval)를 작성하여 특정 활용 사례에 대한 맞춤 테스트 생성 가능
- 커스텀 평가도 레지스트리에 통합하여 활용 가능

개발자들은 OpenAI Evals를 활용해 특정 작업(예: 도메인별 Q&A)에 대해 다양한 모델을 동일한 테스트 세트로 평가하고 비교합니다. 모델 신규 버전 출시 시 이전 버전과의 성능 차이를 검증하거나, 미세조정된 모델이 기존 성능을 유지하는지 확인하는 등 회귀 검사 용도로도 활용됩니다.



### Ragas

Ragas는 Retrieval-Augmented Generation(RAG) 파이프라인 평가에 특화된 오픈소스 프레임워크입니다.
이름 에서 알 수 있듯이, 모델 그 자체 보다는 RAG 시스템의 성능을 평가하는데 특화되어 있습니다.  

- 문서 검색 후 답변을 생성하는 QA 시스템의 성능을 정량화하기 위해 설계
- 다섯 가지 핵심 지표 제공:
  - Faithfulness (충실도)
  - Contextual Relevancy (문맥 관련성)
  - Answer Relevancy (답변 적합성)
  - Contextual Recall (문맥 재현율)
  - Contextual Precision (문맥 정확도)
- 최신 RAG 연구에 기반한 종합적인 평가 지표 제공

Ragas는 RAG 기반의 지식 검색 챗봇이나 문서 QA 시스템에서 많이 활용됩니다. 예를 들어, 사내 데이터베이스 질의응답 시스템에 적용하면 모델의 답변이 주어진 문서에 얼마나 충실한지, 문서 검색이 얼마나 적절했는지를 수치로 모니터링할 수 있습니다. 다만, Ragas의 지표 이름들이 직관적이지 않아 점수가 낮을 때 구체적인 문제 파악을 위해 추가적인 분석이 필요할 수 있습니다.