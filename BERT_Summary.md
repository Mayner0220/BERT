# BERT

## BERT - Bidirectional Encoder Representations form Transformer

- 이 논문은 "Attention is all you need(Vaswani et al., 2017)”에서 소개한 Transformer 구조를 활용한 Language Representatation에 관한 논문입니다.
- BERT는 기본적으로 wiki나 book data와 같은 대용량 unlabeled data으로 모델을 미리 학습 시킨 후, 특정 task을 가지고 있는 labled data으로 transfer learning을 하는 모델입니다.

## API Summary

- Source: https://www.tensorflow.org/api_docs/python
- version: Tensorflow Core v2.2.0
- Langauge: Python

### creat_pretraining_data.py

- tf.complat.v1.flags.Flag (tf.flag)

  ```python
  tf.compat.v1.flags.Flag(
      parser, serializer, name, default, help_string, short_name=None, boolean=False,
      allow_override=False, allow_override_cpp=False, allow_hide_cpp=False,
      allow_overwrite=True, allow_using_method_names=False
  )
  ```

  - Methods

    1. ```python
       flag_type()
       ```

       Returns a str that describes the type of the flag.

    2. ```python
       parse(
       argument
       )
       ```

       Parses string and sets flag value.

    3. ```python
       serialize()
       ```

       Serializes the flag.

    4. ```python
       unparse()
       ```

    5. ```python
       __eq__(
           other
       )
       ```

       Return self==value.

    6. ```python
       __ge__(
           other, NotImplemented=NotImplemented
       )
       ```

       Return a >= b. Computed by @total_ordering from (not a < b).

    7. ```python
       __gt__(
           other, NotImplemented=NotImplemented
       )
       ```

       Return a > b. Computed by @total_ordering from (not a < b) and (a != b).

    8. ```python
       __le__(
           other, NotImplemented=NotImplemented
       )
       ```

       Return a <= b. Computed by @total_ordering from (a < b) or (a == b).

    9. ```python
       __lt__(
           other
       )
       ```

       Return self<value.

