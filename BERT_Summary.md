# BERT - Bidirectional Encoder Representations form Transformer

Source: [https://mino-park7.github.io/nlp/2018/12/12/bert-%EB%85%BC%EB%AC%B8%EC%A0%95%EB%A6%AC/?fbclid=IwAR3S-8iLWEVG6FGUVxoYdwQyA-zG0GpOUzVEsFBd0ARFg4eFXqCyGLznu7w#35-fine-tuning-procedure](https://mino-park7.github.io/nlp/2018/12/12/bert-논문정리/?fbclid=IwAR3S-8iLWEVG6FGUVxoYdwQyA-zG0GpOUzVEsFBd0ARFg4eFXqCyGLznu7w#35-fine-tuning-procedure)

---

## Abstract

### BERT란 무엇인가?

- 이 논문은 "Attention is all you need(Vaswani et al., 2017)”에서 소개한 Transformer 구조를 활용한 Language Representatation에 관한 논문입니다.
- BERT는 기본적으로 wiki나 book data와 같은 대용량 unlabeled data으로 모델을 미리 학습 시킨 후, 특정 task을 가지고 있는 labled data으로 transfer learning을 하는 모델입니다.

### BERT의 이전 모델들

- BERT 이전에, 대용량 unlabeled corpus를 통해 language model을 학습하고 이를 토대로 뒤 쪽에 특정 task를 처리하는 network를 붙이는 방식이 존재하였다.

  Ex) ELMo, OpenAI GT

- 하지만 BERT 논문에서 이전의 모델의 접근 방식은 shallow bidirectional 또는 unidirectional하여 language representation이 부족하다고 표현하였다.

### BERT의 새로운 점

- BERT는 특정 task를 처리하기 위해 새로운 network를 붙일 필요 없이, BERT 모델 자체의 fine-tuning을 통해 해당 task의 state-of-the-art를 달성했다고 한다.

---

## Introduction

### Language model pre-training

- Language model pre-training은 여러 NLP task의 성능을 향상시키는데에 탁월한 효과가 있다고 알려져 있다.

- 이러한 NLP task는 token-level task인 Named Entity Recognition(NER)에서부터 SQuAD question answering task와 같은 task까지 광범위한 부분을 커버한다.

- pre-trained language representation을 적용하는 방식으로 feature-based와 fine-tuning 방식이 존재한다.

  - feature-based: 특정 task를 수행하는 network에 pre-trained language representation을 추가적인 feature로 제공, 두 개의 network를 붙여서 사용한다고 생각하면 된다.

    Ex) ELMo

  - fine-tuning approach: task-specific한 parameter를 최대한 줄이고, pre-trained된 parameter들을 downstream task 학습을 통해 조금만 바꿔주는 방식이다.

    Ex) Generative Pre-trained Transformer(OpenAI GPT)

- ELMo, OpenAI GPT는 pre-training시에 동일한 objective funtion으로 학습을 수행하는 반면, BERT는 새로운 방식으로 pre-trained Language Representation을 학습했고 이는 매우 효과적이였다.

![그림1. BERT, GPT, ELMo (출처 : BERT 논문)](https://mino-park7.github.io/images/2018/12/%EA%B7%B8%EB%A6%BC1-bert-openai-gpt-elmo-%EC%B6%9C%EC%B2%98-bert%EB%85%BC%EB%AC%B8.png)

### Masked Language Model(MLM) & next sentence prediction

- BERT의 pre-training의 새로운 방법론은 크게 2가지로 나눌 수 있다.
  - Masked Language Model(MLM)
  - next sentence prediction
- 기존 방법론: ELMo, OpenAI GPT는 일반적인 language model을 사용했다. 앞의 n 개의 단어를 가지고 뒤의 단어를 예측하는 모델을 세우는 것으로, 이는 필연적으로 unidirectional할 수 밖에 없고, 이러한 단점을 극복하기 위해 ELMo에서는 Bi-LSTM으로 양방향성을 가지려고 노력하지만, 굉장히 shallow한 양방향성 (단방향 concat 단방향)만을 가질 수 밖에 없었다.
- Masked Language Model(MLM): input에서 무작위하게 몇개의 token을 mask 시킨 후, 이를 Transformer 구조에 넣어서 주변 단어의 context만을 보고 mask된 단어를 예측하는 모델이다. OpenAI GPT도 Transformer 구조를 사용하지만, 앞의 단어들만 보고 뒷 단어를 예측하는 Transformer decoder구조를 사용한다. 이와 달리 BERT에서는 input 전체와 mask된 token을 한번에 Transformer encoder에 넣고 원래 token 값을 예측하므로 deep bidirectional 하다고 할 수 있다.
- next sentence prediction: 두 문장을 pre-training시에 같이 넣어줘서 두 문장이 이어지는 문장인지 아닌지 맞추는 것으로, pre-training시에는 50:50 비율로 실제로 이어지는 두 문장과 랜덤하게 추출된 두 문장을 넣어줘서 BERT가 맞추게한다. 이러한 task는 실제 Natural Language Inference와 같은 task를 수행할 때 도움이 된다.

---

## BERT

- BERT의 아키텍처는 Attention is all you need에서 소개된 Transformer를 사용하지만, pre-training과 fiine-tuning시의 아키텍처를 조금 다르게하여 Transfer Learning을 용이하게 만드는 것이 핵심이다.

### Model Architecture

- BERT는 transformer 중에서도 encoder 부분만을 사용한다.
- BERT는 모델의 크기에 따라 base 모델과 large 모델을 제공한다.
  - BERT_base : L=12, H=768, A=12, Total Parameters = 110M
  - BERT_large : L=24, H=1024, A=16, Total Parameters = 340M
  - L : transformer block의 layer 수, H : hidden size, A : self-attention heads 수, feed-forward/filter size = 4H
- 여기서 BERT_base 모델의 경우, OpenAI GPT모델과 hyper parameter가 동일하다. 여기서 BERT의 저자가 의도한 바는 모델의 하이퍼 파라미터가 동일하더라도, pre-training concept를 바꾸어 주는 것만으로 훨씬 높은 성능을 낼 수 있다는 것을 보여주고자 하는 것 같다.

### Input Representation

![그림2. bert input representation (출처: BERT 논문)](https://mino-park7.github.io/images/2019/02/bert-input-representation.png)

- BERT의 input은 3가지 embedding 값의 합으로 이루어져 있다.
- WordPiece embedding을 사용한다. BERT english의 경우 30000개의 token을 사용했다.
- Position embedding을 사용한다. 이는 Transformer에서 사용한 방식과 같음을 알 수 있다.
- 모든 sentence의 첫번째 token은 언제나 `[CLS]`(special classification token) 이다. 이 `[CLS]` token은 transformer 전체층을 다 거치고 나면 token sequence의 결합된 의미를 가지게 되는데, 여기에 간단한 classifier를 붙이면 단일 문장, 또는 연속된 문장의 classification을 쉽게 할 수 있게 된다. 만약 classification task가 아니라면 이 token은 무시하면 된다.
- Sentence pair는 합쳐져서 single sequence로 입력되게 된다. 각각의 Sentence는 실제로는 수 개의 sentence로 이루어져 있을 수 있다(eg. QA task의 경우 `[Question, Paragraph]`에서 Paragraph가 여러개의 문장). 그래서 두 개의 문장을 구분하기 위해, 첫째로는 `[SEP]` token 사용, 둘째로는 Segment embedding을 사용하여 앞의 문장에는 `sentence A embedding`, 뒤의 문장에는 `sentence B embedding`을 더해준다. (모두 고정된 값)
- 만약 문장이 하나만 들어간다면 `sentence A embedding`만을 사용한다.

