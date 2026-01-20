# Transformer from Scratch (Attention Is All You Need 구현)

본 프로젝트는 Vaswani et al.(2017)의 논문 **“Attention Is All You Need”**를 기반으로,
PyTorch의 `nn.Transformer`를 사용하지 않고 Transformer의 핵심 구성 요소를 직접 구현하고,
실제 데이터셋을 이용한 문장 분류 실험을 수행하는 것을 목표로 한다.

---

## 1. Transformer 구조 개요

Transformer는 기존 RNN/LSTM 기반 시퀀스 모델과 달리,
순환 구조 없이 **Self-Attention 메커니즘만으로 문맥 정보를 처리**하는 모델이다.

### Encoder 구조

본 프로젝트에서는 Encoder 구조만을 사용하여 문장 분류 모델을 구성하였다.

각 Encoder Layer는 다음과 같은 순서로 구성된다.

1. Multi-Head Self-Attention  
2. Add & Layer Normalization (Residual Connection)  
3. Position-wise Feed Forward Network  
4. Add & Layer Normalization (Residual Connection)

최종 Encoder는 동일한 Encoder Layer를 여러 층 쌓은 구조이다.

### Decoder 구조

Transformer의 Decoder는 다음 세 가지 하위 층으로 구성된다.

1. Masked Multi-Head Self-Attention  
2. Encoder–Decoder Attention (Cross Attention)  
3. Position-wise Feed Forward Network  

첫 번째 Self-Attention에서는 미래 토큰을 보지 못하도록 causal mask를 적용하여
현재 시점 이전의 토큰 정보만 사용할 수 있게 한다.

두 번째 Attention에서는 Query는 Decoder에서,  
Key와 Value는 Encoder 출력에서 가져와
입력 문장과 출력 문장 간의 정렬(alignment)을 학습한다.

이를 통해 Decoder는 이전에 생성된 토큰과 입력 문맥을 동시에 고려하여
다음 토큰을 생성할 수 있다.


---

## 2. 왜 이러한 구조가 필요한가?

### Self-Attention

Self-Attention은 각 토큰이 문장 내의 모든 토큰을 직접 참조할 수 있도록 하여,
장거리 의존 관계를 효율적으로 모델링할 수 있다.

RNN과 달리 순차적으로 정보를 전달할 필요가 없기 때문에
병렬 처리가 가능하고 학습 속도가 빠르다.

---

### Multi-Head Attention

하나의 Attention만 사용할 경우 관계 표현이 제한될 수 있다.
Multi-Head Attention은 여러 개의 attention head를 사용하여
각기 다른 하위 공간(subspace)에서 토큰 간 관계를 학습할 수 있도록 한다.

이를 통해 문법적 관계, 의미적 관계 등 다양한 패턴을 동시에 포착할 수 있다.

---

### Positional Encoding

Attention 메커니즘 자체에는 순서 정보가 포함되어 있지 않기 때문에,
토큰의 위치 정보를 별도로 제공해야 한다.

논문에서 제안한 사인/코사인 기반 Positional Encoding은
각 위치에 고유한 패턴을 부여하고,
상대적 거리 정보를 선형적으로 추론할 수 있도록 설계되었다.

본 프로젝트에서는 논문에서 제시한 **sin/cos 방식의 Positional Encoding**을 그대로 구현하였다.

---

### Feed Forward Network (FFN)

Attention을 통해 토큰 간 정보가 섞인 후,
각 위치별로 비선형 변환을 적용하여 표현력을 증가시키기 위해
Position-wise Feed Forward Network를 사용한다.

FFN은 모든 위치에 동일한 MLP를 적용하는 구조이다.

---

### Residual Connection & Layer Normalization

Residual Connection은 깊은 네트워크에서도 gradient 흐름을 안정화시키고,
Layer Normalization은 각 층의 분포를 안정적으로 유지하여 학습을 쉽게 만든다.

Transformer의 안정적인 학습을 위해 필수적인 구조이다.

---

## 3. RNN / LSTM과의 구조적 차이점

| 항목 | RNN / LSTM | Transformer |
|--------|------------|------------|
| 처리 방식 | 순차 처리 | 전체 시퀀스 병렬 처리 |
| 장거리 의존성 | 어려움 | Attention으로 직접 연결 |
| 학습 속도 | 느림 | 빠름 |
| 문맥 처리 | hidden state 전달 | attention 가중합 |

Transformer는 순환 구조 없이도 모든 토큰 간 관계를 attention으로 직접 계산하기 때문에,
문맥 정보를 효과적으로 처리할 수 있다.

---

## 4. 구현 내용

다음 구성 요소를 PyTorch로 직접 구현하였다.

- Positional Encoding (sin / cos)
- Scaled Dot-Product Attention
- Self-Attention
- Multi-Head Attention
- Position-wise Feed Forward Network
- Residual Connection & Layer Normalization
- Encoder Layer
- Encoder Stack
- Encoder-only Transformer Classifier

`nn.Transformer` 또는 고수준 Transformer API는 사용하지 않았다.

---

## 5. 실험: AG_NEWS 문장 분류

### 데이터셋

- AG_NEWS (4-class 뉴스 분류 데이터셋)
- 클래스: World, Sports, Business, Sci/Tech

### 전처리

- 소문자 변환
- 공백 기반 토큰화
- 학습 데이터 기반 vocabulary 구성
- Padding / Truncation 적용

---

### 하이퍼파라미터

| 항목 | 값 |
|--------|------|
| max_len | 128 |
| vocab_size | 20,000 |
| num_layers | 2 |
| d_model | 64 |
| num_heads | 4 |
| d_ff | 256 |
| batch_size | 32 |
| optimizer | Adam |
| learning rate | 3e-4 |
| epochs | 3 |

---

### 실험 결과

| Epoch | Train Accuracy | Test Accuracy |
|--------|----------------|---------------|
| 1 | 73.6% | 85.4% |
| 2 | 85.6% | 87.6% |
| 3 | 88.1% | **89.3%** |

학습이 안정적으로 수렴하였으며,
간단한 토크나이징 방식에도 불구하고 높은 분류 성능을 보였다.

---

## 6. Transformer 확장 및 응용

| 모델 | 구조 | 특징 |
|--------|------|--------|
| BERT | Encoder-only | 양방향 문맥 이해, MLM 사전학습 |
| GPT | Decoder-only | 자기회귀 텍스트 생성 |
| ViT | Encoder + Patch Embedding | 이미지 패치를 토큰처럼 처리 |

Transformer는 Attention 구조가 입력 형태에 의존하지 않기 때문에,
텍스트뿐만 아니라 이미지, 음성 등 다양한 도메인으로 확장 가능한 범용 아키텍처로 발전하였다.

---

## 7. 결론

본 프로젝트를 통해 Transformer의 핵심 아이디어인
Self-Attention과 병렬 처리 구조가 어떻게 문맥 정보를 효과적으로 처리하는지 이해할 수 있었다.

또한 실제 데이터셋을 활용한 실험을 통해,
직접 구현한 Transformer Encoder가 실용적인 성능을 낼 수 있음을 확인하였다.
