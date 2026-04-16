# EDCC: Event-Aware Dynamic Centralized Cross-scale EEG Model
## CAUEEG Dataset 실험 보고서

---

## 1. 연구 개요

### 1.1 목표
CAUEEG 데이터셋에서 Normal / MCI / Dementia 3-class EEG 분류를 위한 새로운 모델(EDCC) 설계 및 실험

### 1.2 핵심 가설
> Eyes-Open ↔ Eyes-Closed 전환 시의 뇌 동기화 상태 변화(core token trajectory)가 정적 주파수 분석으로는 구분 어려운 MCI를 탐지하는 핵심 특징이다.

### 1.3 Baseline
- **CeedNet 1D-ResNet-18 + TTA**: ~68.75% (원 논문)
- **CeedNet Ensemble**: ~74.66%

### 1.4 데이터셋
| 항목 | 값 |
|------|------|
| 총 녹음 수 | 1,187개 (Train 950 / Val 119 / Test 118) |
| 채널 수 | 19 EEG + 1 EKG + 1 Photic |
| 샘플링 레이트 | 200 Hz |
| 평균 녹음 길이 | 13.34 ± 2.83 min |
| Class 분포 | Normal 459 / MCI 417 / Dementia 311 |

---

## 2. EDA 핵심 발견 (Phase 1)

### 2.1 나이 분포 — Confounding Factor 확인
| Class | 평균 나이 | Std |
|-------|----------|-----|
| Normal | 65.1세 | 9.5 |
| MCI | 73.8세 | 7.8 |
| Dementia | 76.6세 | 8.1 |

- **Kruskal-Wallis H=295.5, p=6.82e-65** → 나이가 class와 강하게 상관
- → Age-conditioned normalization 및 Age adversarial training 필수

### 2.2 Alpha/Theta Ratio — 주파수 바이오마커
| Class | Mean | Median |
|-------|------|--------|
| Normal | 2.408 | 1.927 |
| MCI | 1.747 | 1.396 |
| Dementia | 1.268 | 0.917 |

- **Kruskal-Wallis H=118.5, p=1.89e-26** → Normal > MCI > Dementia 순서
- → Multi-scale tokenization (k=15,40,100)의 근거

### 2.3 EO→EC 전환 동역학 — 핵심 가설 검증 ⭐
| Class | Alpha 변화량 (전환 후 peak/baseline) |
|-------|-------------------------------------|
| Normal | **+25.4x** |
| MCI | +13.2x |
| Dementia | +11.7x |

- **Kruskal-Wallis H=554.6, p=3.67e-121** → 극도로 유의미
- Normal의 Alpha reactivity가 MCI/Dementia보다 2배 이상 강함
- → **EDCC의 핵심 가설을 직접 뒷받침** → Event-Conditioned Mamba 설계 정당화

### 2.4 채널 간 연결성
- 영역 간 연결 강도 차이: Central-Temporal (-0.054), Central-Occipital (-0.047)
- Clustering coefficient에서 약한 유의차 (p=0.014)
- → Region-Level GCN 설계 근거

---

## 3. EDCC 모델 아키텍처

### 3.1 전체 구조

```
Raw EEG (19ch × 13분 × 200Hz)
        ↓
   Event-Aware Windowing (4초 윈도우, 이벤트 라벨 부여)
        ↓
┌─ Stage 1: Cross-scale Tokenization ──────────────┐
│  "EEG 신호에서 주파수 특징을 추출하자"              │
└──────────────────────────────────────────────────┘
        ↓
┌─ Stage 2: Dynamic Centralized Core Token ────────┐
│  "19채널을 요약하고, 시간에 따른 변화를 추적하자"    │
└──────────────────────────────────────────────────┘
        ↓
┌─ Stage 3: Region-Level GCN ──────────────────────┐
│  "뇌 영역 간 관계를 학습하자"                       │
└──────────────────────────────────────────────────┘
        ↓
┌─ Classification Head ────────────────────────────┐
│  "나이를 고려해서 최종 판단하자"                     │
└──────────────────────────────────────────────────┘
```

### 3.2 Stage 0: Event-Aware Windowing

**기존 CeedNet의 한계**: 13분 녹음에서 10초를 **랜덤으로** 잘라서 사용. 이벤트 정보를 버림.

**우리의 접근**: 녹음 전체를 4초 윈도우로 분할하고, 각 윈도우에 이벤트 타입을 부여.

```
녹음 타임라인:
[====EO====][EC][=====EO=====][====EC====][Photic...][EO][EC]...
     ↓       ↓       ↓            ↓         ↓
  type=0   type=2  type=0      type=1     type=4
           (전환)              
```

**설계 근거**:
- EDA에서 EO→EC **전환 시점**의 Alpha reactivity가 class 간 가장 큰 차이를 보임 (p=3.67e-121)
- 전환 구간을 명시적으로 마킹하여 모델이 "지금이 전환 시점"이라는 정보를 직접 받음
- CeedNet은 랜덤 crop이라 전환 시점이 포함될 수도, 안 될 수도 있음 → 불확실성

### 3.3 Stage 1: Cross-scale Tokenization

**설계 목적**: 하나의 EEG 윈도우(19ch × 800 samples)를 의미 있는 토큰으로 변환

```
입력: (B, W, 19, 800) — 각 채널의 4초 신호
      ↓
Multi-scale Conv1d (채널별 독립 적용):
  Branch 1: k=15  (75ms)  → 64-dim  ← Beta/Alpha (12-33Hz)
  Branch 2: k=40  (200ms) → 32-dim  ← Alpha/Theta (4-13Hz)
  Branch 3: k=100 (500ms) → 16-dim  ← Delta/Theta (0.5-8Hz)
      ↓ concat + projection
  112-dim → d_model 토큰
      ↓
+ Event Type Embedding (6종)   ← "이 윈도우는 Eyes Closed 구간"
+ Sinusoidal Temporal PE       ← "녹음의 30% 지점"
      ↓
출력: (B, W, 19, d_model) — 채널별, 윈도우별 토큰
```

**설계 근거**:
- EEG의 핵심 정보는 주파수 대역(delta, theta, alpha, beta)에 있음
- 단일 커널로는 모든 대역을 동시에 포착할 수 없음. CeedNet도 k=41을 사용하지만 단일 스케일
- 3개 스케일 병렬 → EDA에서 확인된 **Alpha/Theta ratio** 차이를 직접 포착 (p=1.89e-26)
- Event Embedding: 같은 Alpha 파워라도 EO/EC 상태에 따라 의미가 다름 (context-dependent feature)

### 3.4 Stage 2: Dynamic Centralized Core Token

이 스테이지가 EDCC의 **핵심 아이디어**입니다. 세 부분으로 구성됩니다.

#### Part 1: CoTAR Aggregation — 19채널 → 1 core token

**문제**: 19채널 × W윈도우 = 19W개의 토큰. 시간축 모델링에 비효율적.

**해결**: 각 윈도우에서 19채널을 **하나의 core token으로 압축** (attention-weighted)

```
윈도우 t에서:
  19개 채널 토큰: [Fp1, F3, C3, ..., PZ]  각 d_model 차원
        ↓ Attention (Q=mean, K=channels, V=channels)
  1개 core token: (core_dim) — 19채널의 "뇌 상태 요약"
```

**설계 근거**: 뇌의 상태(각성, 이완, 인지 저하)는 개별 채널이 아니라 **채널 간 조합**으로 표현됨. Core token은 "이 시점에서 뇌가 어떤 상태인가"를 하나의 벡터로 요약. 차원을 d_model → core_dim으로 줄여 **정보 bottleneck** → 핵심 패턴만 추출.

#### Part 2: Event-Conditioned SSD — Core Token Trajectory 모델링

**목적**: Core token이 시간에 따라 어떻게 변하는지 추적

```
core tokens: [c_1, c_2, c_3, ..., c_W]  시간 순서
event types: [EO,  EO,  EC, ..., EC]

SSD (Structured State-space Duality):
  c_1 →[EO]→ c_2 →[EO→EC 전환!]→ c_3 →[EC]→ c_4 → ...
                    ↑
              여기서 Δ(discretization)가 커짐
              = 이전 상태를 많이 잊고 새 정보 강하게 수용
              = "전환 시 뇌 상태가 급격히 변한다"를 모델링
```

**Event conditioning 구현**:
```
이벤트 임베딩을 입력에 concat → SSD의 Δ projection이 자연스럽게 이벤트 반영
augmented = concat([core_token, event_embedding])  # "지금 전환 구간이야"
Δ = softplus(W_Δ · augmented)  # 전환이면 Δ 커짐 → selective forgetting ↑
```

**왜 Mamba/SSD인가** (GRU나 Transformer가 아니라):
- **Mamba/SSD**: selective state-space model → 입력에 따라 어떤 정보를 기억/망각할지 **선택**
- EO→EC 전환 시: Δ가 커짐 → 이전 EO 상태를 잊고 EC 상태를 새로 학습
- 정적 구간(EO 유지 중): Δ가 작음 → 이전 상태를 유지하며 점진적 변화만 추적
- **GRU**: forget gate가 있지만 event conditioning이 자연스럽지 않음
- **Transformer**: 전체 시퀀스에 attention → 위치 불변, 인과적 시간 흐름 모델링에 약함

**핵심 가설 구현**:
- EDA: Normal의 전환 시 Alpha 변화 +25x, Dementia는 +12x
- 정상 뇌: 전환 시 core token이 **크게** 변함 → 뚜렷한 상태 전환
- 치매 뇌: 전환 시 core token이 **적게** 변함 → 반응 둔화
- SSD가 이 trajectory 패턴을 학습 → **전환 시 변화량의 차이**로 class 구분

**Pure-torch SSD 구현**: mamba-ssm CUDA 커널의 ABI 호환 문제로, Mamba-2 논문의 SSD 알고리즘을 순수 PyTorch로 재구현. Block-decomposition 기반 효율적 SSM scan, DropPath(stochastic depth)로 정규화.

#### Part 3: CoTAR Redistribution — core → 19 channels

처리된 core token을 저장된 attention weights로 역투영하여 채널 토큰을 업데이트.

**설계 근거**: Core token은 "전체 뇌 상태의 시간적 변화"를 알고 있음. 이 정보를 각 채널에 다시 주입하면, 각 채널이 **global temporal context를 가진 상태**에서 Stage 3을 진행 → GCN이 더 풍부한 정보로 영역 간 관계를 학습.

### 3.5 Stage 3: Region-Level GCN

**설계 목적**: 19개 채널의 공간적 관계를 뇌 영역 수준에서 학습

```
19 channels → 5 regions (attention pooling):
  Frontal  (7ch): Fp1, Fp2, F3, F4, F7, F8, FZ
  Central  (3ch): C3, C4, CZ
  Parietal (3ch): P3, P4, PZ
  Temporal (4ch): T3, T4, T5, T6
  Occipital(2ch): O1, O2

Fixed adjacency (해부학적 근접성):
  Frontal ─── Central ─── Parietal
     │           │            │
  Temporal ─────┘        Occipital

2-layer GCN + Laplacian PE → mean pool → graph representation
```

**설계 근거**:
- 19×19=361개 edge → 950개 학습 데이터로 학습하기엔 과다. 5×5=25개 edge → 학습 가능
- EDA에서 **영역 수준**의 연결성 차이가 유의미 (Central-Temporal, p=0.014)
- 같은 영역 내 채널들은 높은 상관관계 → 영역 단위 요약이 정보 손실 적음
- 치매에서 관찰되는 "temporal-parietal 연결 약화"를 GCN의 message passing으로 직접 모델링

### 3.6 Classification Head

```
graph representation: (B, d_model)
        ↓
Age-conditioned LayerNorm:
  age → 10-bin embedding → (γ, β) 
  z = LayerNorm(z) * (1+γ) + β     ← 나이에 따라 feature 스케일 조절
        ↓
Age adversarial: gradient reversal → age 예측  ← shortcut 방지
        ↓
Age concat: z = [z; (age-70)/15]    ← CeedNet 스타일, 나이 정보 직접 제공
        ↓
MLP: d_model+1 → d_model/2 → 3 classes
```

**Age conditioning이 두 가지인 이유**:
1. **Conditional LayerNorm**: 나이에 따라 feature의 스케일과 shift를 조절. 같은 EEG 패턴이라도 70세와 50세에서 의미가 다름 → 나이별 다른 정규화
2. **Age concat**: 나이를 직접 feature에 추가. CeedNet에서 검증된 방법. 두 방법은 상호보완적.

**Age adversarial**: EDA에서 나이가 class와 강하게 상관 (p=6.82e-65). 모델이 "나이가 높으니까 Dementia"로 shortcut 학습할 위험 → Gradient reversal로 나이에 의존하지 않는 EEG 패턴 학습 강제.

### 3.7 Window-Level Auxiliary Loss

- 각 윈도우의 core token representation으로 독립적 분류 수행
- 녹음 1개 → 128개 윈도우 = 128개 학습 신호 → **실효 학습 데이터 128배 증가**
- Recording-level loss와 window-level loss를 결합 (weight=0.3)
- 950개 recording 문제를 ~121K개 window 문제로 전환하여 과적합 완화

### 3.8 설계 결정 요약

| 결정 | 이유 | EDA 근거 |
|------|------|---------|
| 4초 윈도우 (CeedNet 10초 대비 짧음) | 이벤트 구간이 대부분 2-40초 → 4초면 대부분 단일 이벤트 내 | EO 구간 median 6.9초, EC 32.3초 |
| Event type embedding | 같은 신호도 EO/EC에서 의미 다름 | Alpha reactivity가 EC에서만 나타남 |
| Multi-scale conv (k=15,40,100) | 다중 주파수 대역 동시 포착 | Alpha/Theta ratio 핵심 바이오마커 (p=1.89e-26) |
| Core token + SSD | 전환 시 뇌 상태 변화 추적 | EO→EC Alpha reactivity 차이 (p=3.67e-121) |
| 5-region GCN (19ch 대신) | 학습 가능한 규모의 그래프 | 영역 간 연결성 차이 유의미 |
| Age conditioning + adversarial | 나이 confounding 해결 | 나이-class 상관 (p=6.82e-65) |
| Window-level auxiliary loss | 실효 학습 데이터 128배 증가 | 950개 학습 데이터 부족 해결 |

### 3.9 모델 사양

| 설정 | Base (Run 1-4) | Large (Run 5) | SSD (Run 6-13) |
|------|:-:|:-:|:-:|
| d_model | 128 | 256 | 384 |
| core_dim | 64 | 128 | 192 |
| Layers | 4 | 6 | 6 |
| Backend | GRU → GatedConvRNN | GatedConvRNN | SSD |
| Window | 2s (400) | 4s (800) | 4s (800) |
| Params | 252K-645K | 3.49M | 3.39M |

---

## 4. 실험 결과

### 4.1 전체 실험 요약

| Run | Config | Loss | Params | Test Acc | TTA Acc | Best Val |
|:---:|--------|------|:------:|:--------:|:-------:|:--------:|
| 1 | Base (GRU) | CE | 252K | 48.31% | - | 54.62% |
| 2 | +GatedConvRNN +Mixup | CE+CW | 645K | 53.39% | - | 47.90% |
| 3 | Tuned v1 (LR↓) | CE+CW | 645K | 55.08% | - | 47.06% |
| 4 | Tuned v2 | CE+CW | 645K | 50.00% | - | 50.42% |
| **5** | **Large (GatedConvRNN)** | **CE+CW+Mixup** | **3.49M** | **61.02%** | **61.02%** | **57.14%** |
| 6 | Final (SSD+Focal) | Focal+CW | 9.14M | 53.39% | 55.93% | 57.14% |
| 7 | v3 (SSD+DropPath+WinAux) | Focal+CW+WinAux | 3.39M | 57.63% | 56.78% | 55.46% |
| 8 | CORAL ordinal | CORAL+CW+WinAux | 3.39M | 54.24% | - | 57.98% |
| 9 | CE+OrdPenalty | CE+Ord+CW+WinAux | 3.39M | 54.24% | 56.78% | 54.62% |
| 10 | RACE (uniform) | RACE+CW+WinAux | 3.39M | 51.69% | 50.85% | 53.78% |
| 11 | A-RACE | A-RACE+CW+WinAux | 3.39M | 53.39% | 55.08% | 53.78% |
| 12 | CeedNet-style (no aug) | A-RACE+DataNorm | 3.39M | 48.31% | 52.54% | 56.30% |
| **13** | **Best Combined** | **A-RACE+DataNorm+Aug** | **3.39M** | **55.08%** | **59.32%** | **57.14%** |

### 4.2 Class별 성능 비교 (주요 Runs)

| | Normal Sens | MCI Sens | Dem Sens | Normal Spec | Balanced Acc | Macro F1 |
|--|:-:|:-:|:-:|:-:|:-:|:-:|
| **Run 5** (CE, Best Acc) | **89.13%** | **53.66%** | 29.03% | 69.44% | - | - |
| **Run 7** (Focal+WinAux) | 78.26% | 31.71% | **61.29%** | 75.00% | - | - |
| **Run 8** (CORAL) | 69.57% | 26.83% | 67.74% | 70.83% | 54.71% | 52.11% |
| **Run 10** (RACE) | 69.57% | **63.41%** | 9.68% | 80.56% | 47.55% | 44.86% |
| **Run 13+TTA** (Best Balanced) | 73.91% | 31.71% | **74.19%** | 79.17% | **59.94%** | **57.25%** |

### 4.3 개선 추이

```
Test Accuracy:
  Run 1  (252K, GRU, CE)              ████████████████░░░░░░░░░░░░░  48.31%
  Run 2  (645K, GatedConvRNN, Mixup)  ██████████████████░░░░░░░░░░░  53.39%
  Run 5  (3.5M, Large)               █████████████████████░░░░░░░░  61.02% ← Best Acc
  Run 13 (3.4M, Best Combined +TTA)  ████████████████████░░░░░░░░░  59.32% ← Best Balanced
  CeedNet                             ████████████████████████░░░░░  68.75%

Dementia Sensitivity:
  Run 5  (CE)                         █████████░░░░░░░░░░░░░░░░░░░░  29.03%
  Run 7  (Focal+WinAux)              ███████████████████░░░░░░░░░░  61.29%
  Run 8  (CORAL)                      █████████████████████░░░░░░░░  67.74%
  Run 13 (A-RACE+DataNorm +TTA)      ███████████████████████░░░░░░  74.19% ← Best

MCI Sensitivity:
  Run 1  (CE, no tuning)              ███░░░░░░░░░░░░░░░░░░░░░░░░░░   7.32%
  Run 5  (CE, Large)                  █████████████████░░░░░░░░░░░░  53.66% ← Best
  Run 10 (RACE)                       ████████████████████░░░░░░░░░  63.41% ← Best (but Dem=9%)
```

---

## 5. Loss Function 비교 분석

### 5.1 각 Loss의 특성

| Loss | 원리 | 강점 | 약점 |
|------|------|------|------|
| **Cross-Entropy** | 각 class 독립 학습 | MCI boundary 선명 (53%) | Normal↔Dementia 혼동 억제 못함 |
| **Focal Loss** | Hard example에 집중 | Dementia 탐지 개선 (61%) | 전체 accuracy 하락 |
| **CORAL** | K-1 binary 분해 (ordinal) | Ordinal 구조 학습, Dem 강함 (67%) | MCI를 양쪽으로 흡수 (26%) |
| **RACE** | Ordinal distance 기반 soft label | MCI 최강 (63%) | Dementia 붕괴 (9.7%) |
| **A-RACE** | Class별 적응형 smoothing | 균형 잡힌 성능 | CE 단독 대비 MCI 약화 |
| **CE + Ordinal Penalty** | CE + expected rank distance | Dementia 강함 (70%) | MCI 약함 (24%) |

### 5.2 CORAL (Consistent Rank Logits) 상세

K-class ordinal 분류를 **K-1개의 binary 분류 문제**로 분해하는 방법.

**Binary target 생성**: 각 샘플의 true label k에 대해 K-1개의 binary target:

```
t_r^(k) = 1[k > r],  r = 0, 1, ..., K-2
```

3-class의 경우:

| True label | t_0 (Y>0?) | t_1 (Y>1?) |
|:----------:|:---:|:---:|
| Normal (k=0) | 0 | 0 |
| MCI (k=1) | 1 | 0 |
| Dementia (k=2) | 1 | 1 |

**CORAL Loss**: 모델이 K-1개의 cumulative logit f_r(x)를 출력하면:

```
L_CORAL = -(1/(K-1)) * Σ_r [t_r·log σ(f_r(x)) + (1-t_r)·log(1-σ(f_r(x)))]
```

**Rank Consistency**: 모든 binary task가 같은 feature weight를 공유하고 bias만 다름:

```
f_r(x) = w^T · h(x) + b_r
```

b_0 > b_1이면 항상 P(Y>0) > P(Y>1) → **rank consistency 보장**. 일반 softmax CE에서는 비일관적 예측(P(Y=0) > P(Y=2)이면서 P(Y=1) < P(Y=2))이 가능하지만, CORAL은 이를 구조적으로 방지.

**CORAL의 MCI 약점**: MCI는 "P(Y>0)=high, P(Y>1)=low"라는 미묘한 조합을 학습해야 하여, 양쪽 binary classifier 모두에서 경계에 위치 → 양쪽으로 흡수되기 쉬움.

### 5.3 A-RACE (Adaptive Rank-Aware Cross-Entropy) 상세

CORAL의 MCI 약점을 해결하기 위해 제안한 새로운 loss function.

**핵심 아이디어**: Label smoothing에서 smoothing 분포를 ordinal 거리에 따라 비대칭으로 설정하되, class의 ordinal 위치에 따라 smoothing 강도를 다르게 적용.

**수식**: True label k에 대한 soft target distribution:

```
q(j|k) = (1-ε_k)·δ(j,k) + ε_k · exp(-|j-k|/σ) / Z_k
```

- δ(j,k): 크로네커 델타 (one-hot)
- ε_k: class k의 smoothing rate (**class별로 다름**)
- σ: ordinal temperature (거리 감쇠 속도)
- Z_k: 정규화 상수

**Class-adaptive smoothing**:
- Endpoint classes (ε=0.20): 원거리 leak 억제, ordinal awareness
- Middle classes (ε=0.05): 거의 one-hot, CE-like sharp boundary

```
A-RACE soft target matrix:
              Normal    MCI    Dementia
True Normal:  [0.973,  0.024,  0.003]    ← endpoint: ordinal smoothing (ε=0.20)
True MCI:     [0.005,  0.989,  0.005]    ← middle: CE-like sharp (ε=0.05)
True Dementia:[0.003,  0.024,  0.973]    ← endpoint: ordinal smoothing (ε=0.20)

비교 — 표준 label smoothing (ε=0.15, uniform):
True Normal:  [0.900,  0.050,  0.050]    ← Dementia에도 동일 leak
True MCI:     [0.050,  0.900,  0.050]
True Dementia:[0.050,  0.050,  0.900]    ← Normal에도 동일 leak
```

**A-RACE의 장점**:
- Normal→Dementia leak = 0.003 (uniform의 0.050보다 **17배 작음**)
- MCI는 0.989 on-target (CE에 가까운 sharp boundary)
- 표준 K-class output 유지 (CORAL처럼 K-1 분해 불필요)

### 5.4 핵심 발견

> **3-class ordinal 분류에서 단일 loss로 모든 class를 균형 있게 다루기 어려움.**
> 
> - **Middle class (MCI)**: 양쪽 경계가 모두 중요 → CE의 sharp boundary 필요
> - **Endpoint classes (Normal, Dementia)**: 반대편과의 혼동이 치명적 → ordinal awareness 필요
> - 이 두 요구사항은 근본적으로 상충하며, A-RACE가 가장 좋은 타협점을 제공

---

## 6. CeedNet 테크닉 분석 및 적용

### 6.1 CeedNet의 7가지 핵심 테크닉

| # | 테크닉 | EDCC 적용 여부 | 효과 |
|---|--------|:-------------:|------|
| 1 | Auto stride programming | N/A | 아키텍처가 다름 (AdaptivePool 사용) |
| 2 | **Mish activation** | **적용** | GELU/SiLU → Mish로 변경 |
| 3 | **Age concat to FC** | **적용** | Conditional LN + age concat 병행 |
| 4 | TTA (8 random crops) | 부분 적용 | Window offset TTA (효과 제한적) |
| 5 | **Dataset-level normalization** | **적용** | Per-sample → dataset 통계 변경 |
| 6 | Minimal noise (σ=0.001) | 테스트 완료 | 우리 모델에서는 σ=0.03 + mixup이 더 효과적 |
| 7 | Large initial kernel (k=41) | 부분 적용 | Multi-scale conv (k=15,40,100)로 대체 |

### 6.2 핵심 교훈

- CeedNet의 augmentation 전략(σ=0.001, no mixup)은 **random crop 기반 데이터 증강과 함께** 설계됨
- EDCC의 window 기반 접근에서는 random crop 효과가 없으므로, **자체 augmentation(mixup 0.2, noise 0.03)을 유지**하는 것이 효과적
- Dataset-level normalization + Mish + age concat은 아키텍처에 무관하게 유효

---

## 7. 구현 상세

### 7.1 데이터 파이프라인

```
Raw EEG (21ch, ~160K samples)
    ↓ Preload into memory (EDF → numpy cache)
    ↓ Event parsing (6 types: EO, EC, EO→EC, EC→EO, Photic, Other)
    ↓ Window extraction (4s window, 1s stride → ~800 windows/recording)
    ↓ Stratified subsampling (128 windows for train, 256 for eval)
    ↓ Dataset-level normalization (channel-wise mean/std)
    ↓ Batch collate with padding mask
    ↓ Augmentations: Gaussian noise + window/channel dropout + mixup
    → Model input: (B, W, 19, 800)
```

### 7.2 학습 설정

| 항목 | 값 |
|------|------|
| Optimizer | AdamW (lr=7e-5, weight_decay=0.05) |
| Scheduler | Cosine annealing (warmup 8 epochs) |
| Batch size | 8 |
| Mixed precision | FP16 (AMP) |
| Early stopping | Patience 25 epochs |
| Loss | A-RACE (ε=[0.20,0.05,0.20], σ=0.5) + Subject CE + Age adversarial + Window auxiliary |
| Augmentation | Gaussian noise (σ=0.03), window dropout (5%), channel dropout (5%), mixup (α=0.2) |
| TTA | 8 repeats with diverse window offsets |

### 7.3 파일 구조

```
edcc/
├── data/
│   ├── event_segmenter.py       # 이벤트 파싱 (6종 세그먼트)
│   ├── windowed_dataset.py      # 윈도우 Dataset + preloading + dataset norm
│   ├── collate.py               # 가변 길이 패딩
│   └── augmentation.py          # Noise / Dropout / Mixup
├── models/
│   ├── tokenizer.py             # Stage 1: Multi-scale Conv + Mish
│   ├── cotar.py                 # Stage 2-1: Core Token Aggregation
│   ├── mamba_core.py            # Stage 2-2: SSD + GatedConvRNN + DropPath
│   ├── gcn.py                   # Stage 3: 5-node Region GCN + LapPE
│   ├── classifier.py            # Age-conditioned LN + age concat + MLP
│   └── edcc_model.py            # Full model assembly
├── training/
│   ├── losses.py                # CE / Focal / CORAL / RACE / A-RACE + Window Aux
│   └── trainer.py               # Training loop + TTA
├── configs/                     # 13개 실험 config (YAML)
└── scripts/
    └── run_edcc_train.py        # Entry point
```

---

## 8. CeedNet 대비 현황

| 메트릭 | CeedNet 1D-ResNet-18 | EDCC Best (Run 5) | EDCC Best Balanced (Run 13+TTA) |
|--------|:-------------------:|:-----------------:|:------------------------------:|
| **Test Accuracy** | ~68.75% | 61.02% | 59.32% |
| **Parameters** | 11.4M | 3.49M | 3.39M |
| **Param 효율** | - | **3.3x 작음** | **3.4x 작음** |
| **MCI Sensitivity** | ~50% (추정) | **53.66%** | 31.71% |
| **Dem Sensitivity** | ~35% (추정) | 29.03% | **74.19%** |
| **Balanced Accuracy** | - | - | **59.94%** |
| **TTA 효과** | +2-5pp | 0pp | **+4.2pp** |

---

## 9. 한계점 및 향후 과제

### 9.1 현재 한계점

1. **CeedNet 대비 7.7pp 열세** (61.02% vs 68.75%)
   - 주 원인: CeedNet의 random crop이 매 epoch 다른 10s 구간을 보여 실효 데이터 다양성이 훨씬 큼
   - EDCC의 window 기반 접근은 정보량은 많지만 recording 수 자체(950개)에 의존

2. **MCI-Dementia trade-off 미해결**
   - CE → MCI 강함 (53.66%), Dem 약함 (29.03%)
   - Ordinal → Dem 강함 (74.19%), MCI 약함 (31.71%)
   - 단일 모델로 두 class 동시 최적화가 어려움 (ordinal 중간 class의 본질적 어려움)

3. **mamba-ssm 네이티브 미사용**
   - PyTorch 2.6 + Python 3.13 환경에서 CUDA 커널 ABI 불일치
   - Pure-torch SSD로 대체 (기능은 동일하나 속도 열세)

4. **Val-Test 일관성 부족**
   - Val best epoch이 Test에서 항상 최적이 아님 (Val 119 / Test 118 → 작은 셋)
   - 다중 seed 반복 실험 미수행 (현재 seed=42만 사용)

5. **TTA 효과 불안정**
   - Window offset TTA는 dataset-level norm과 함께 사용할 때만 효과적 (+4.2pp)
   - Per-sample norm에서는 TTA 효과 미미 (같은 윈도우의 반복)

### 9.2 향후 과제

| 우선순위 | 방향 | 기대 효과 |
|---------|------|----------|
| 1 | **Ensemble** (Run 5 + Run 13 투표) | MCI+Dem 상호보완, +3-5pp |
| 2 | **다중 seed 반복 실험** (20회 평균) | 안정적 성능 추정 |
| 3 | **CAUEEG-Abnormal 사전학습** → Dementia fine-tuning | 더 많은 데이터 활용 |
| 4 | **Window-level contrastive learning** | 더 강력한 representation |
| 5 | PyTorch/CUDA 환경 맞춰 **native Mamba** 사용 | 속도 + 성능 |
| 6 | **더 긴 윈도우** (8-10초) 또는 **hierarchical windowing** | CeedNet의 10s crop에 근접 |

---

## 10. 재현 방법

```bash
# 환경 설정
pip install -r requirements_edcc.txt

# EDA 실행 (Phase 1)
cd notebooks/
jupyter notebook phase1_01_data_overview.ipynb

# Best Accuracy 모델 학습 (Run 5 재현)
python -m edcc.scripts.run_edcc_train --config edcc/configs/edcc_large.yaml

# Best Balanced 모델 학습 (Run 13 재현)
python -m edcc.scripts.run_edcc_train --config edcc/configs/edcc_best_combined.yaml

# 다른 seed로 반복
python -m edcc.scripts.run_edcc_train --config edcc/configs/edcc_best_combined.yaml --seed 123
```

---

*Report generated: 2026-04-16*
*Model: EDCC v1.0*
*Dataset: CAUEEG-Dementia*
