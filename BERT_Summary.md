# BERT - Bidirectional Encoder Representations form Transformer

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



## Introduction

### Language model pre-training

- Language model pre-training은 여러 NLP task의 성능을 향상시키는데에 탁월한 효과가 있다고 알려져 있다.
- 이러한 NLP task는 token-level task인 Named Entity Recognition(NER)에서부터 SQuAD question answering task와 같은 task까지 광범위한 부분을 커버한다.
- 