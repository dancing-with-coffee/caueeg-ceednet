# Dynamic Centralized Cross-scale EEG Model for Dementia Detection
## CAUEEG Dataset 실험 설계서

> **연구 목표**: CAUEEG 데이터셋에서 Normal / MCI / Dementia 3-class 분류를 위한  
> Event-Aware Dynamic Centralized Cross-scale (EDCC) 모델 설계 및 실험  
> **핵심 가설**: Eyes-Open ↔ Eyes-Closed 전환 시의 뇌 동기화 상태 변화(core token trajectory)가  
> 정적 주파수 분석으로는 구분 어려운 MCI를 탐지하는 핵심 특징이다.

---

## 1. CAUEEG 데이터셋 개요

### 1.1 기본 사양

| 항목 | 값 |
|------|------|
| 총 녹음 수 | 1,379개 (1,155명 환자) |
| 채널 수 | 19 EEG + 1 EKG + 1 Photic |
| 전극 배치 | International 10-20 system |
| 전극 위치 | Fp1, F3, C3, P3, O1, Fp2, F4, C4, P4, O2, F7, T3, T5, F8, T4, T6, FZ, CZ, PZ |
| 참조 전극 | Linked earlobe referencing → Common average referencing |
| 샘플링 레이트 | 200 Hz |
| 대역 필터 | 0.5–70 Hz |
| 평균 녹음 길이 | 13.34 ± 2.83 min |
| 데이터 형식 | European Data Format (EDF) |
| 환자 평균 나이 | 70.77 ± 9.90 세 |
| 성비 | 약 60 남 : 100 여 |

### 1.2 CAUEEG-Dementia Task

| Split | Normal | MCI | Dementia | Total |
|-------|--------|-----|----------|-------|
| Training | 367 | 334 | 249 | 950 |
| Validation | 46 | 42 | 31 | 119 |
| Test | 46 | 41 | 31 | 118 |
| **Total** | **459** | **417** | **311** | **1,187** |

### 1.3 이벤트 정보

CAUEEG는 녹음 중 발생한 이벤트 히스토리를 JSON으로 제공:
- Eyes Open / Eyes Closed (각 ~16,800회)
- Photic Stimulation On/Off (3, 6, 9, 12, 15, 18, 21, 24, 27, 30 Hz)
- Paused / Recording Resumed
- 아티팩트 관련: Move, swallowing, artifact

### 1.4 뇌 영역 그룹핑 (19채널 → 5영역)

| 뇌 영역 | 채널 | 기능적 역할 |
|---------|------|----------|
| **Frontal** | Fp1, Fp2, F3, F4, F7, F8, FZ | 인지, 실행 기능, 감정 |
| **Central** | C3, C4, CZ | 운동/감각 |
| **Parietal** | P3, P4, PZ | 공간 인지, 통합 |
| **Temporal** | T3, T4, T5, T6 | 기억, 언어, 청각 |
| **Occipital** | O1, O2 | 시각 처리, Alpha 파 생성 |

---

## 2. 실험 파이프라인

```
Phase 1: EDA (Exploratory Data Analysis)
    ├── 2.1 데이터 로딩 및 기본 통계
    ├── 2.2 이벤트 구조 분석
    ├── 2.3 주파수 대역별 파워 분석 (Class별)
    ├── 2.4 Eyes-Open vs Eyes-Closed 전환 동역학 분석
    └── 2.5 채널 간 연결성 분석 (Class별, 이벤트별)

Phase 2: Baseline 실험
    ├── 2.6 CEEDNet 재현 (baseline 성능 확인)
    └── 2.7 간단한 Foundation Model fine-tuning (LaBraM/BIOT)

Phase 3: EDCC 모델 구현 및 실험
    ├── 2.8 Stage 1: Event-Aware Cross-scale Tokenization
    ├── 2.9 Stage 2: Dynamic Centralized Core Token + Mamba
    ├── 2.10 Stage 3: Region-Level GCN
    └── 2.11 Full Model 통합 학습

Phase 4: 분석 및 해석
    ├── 2.12 Ablation Study
    ├── 2.13 Core Token Trajectory 시각화
    ├── 2.14 이벤트 전환 구간 vs 정적 구간 비교
    └── 2.15 영역 간 연결 패턴 해석
```

---

## 3. Phase 1: Exploratory Data Analysis (EDA)

### 3.1 데이터 로딩 및 기본 통계

**목표**: CAUEEG 데이터셋의 구조 이해 및 품질 확인

```
수행 작업:
1. EDF 파일 로딩 (MNE-Python 사용)
2. 신호 길이 분포 확인 (class별)
3. 나이 분포 확인 (class별)
4. 성별 분포 확인 (class별)
5. 결측 채널/불량 녹음 탐지
6. Task annotation JSON 파싱 및 데이터 매핑
```

**핵심 확인 사항**:
- Normal/MCI/Dementia 간 나이 분포 차이 (confounding factor 확인)
- 녹음 길이의 class별 차이 유무
- EDF 파일과 annotation JSON의 매핑 정합성

### 3.2 이벤트 구조 분석

**목표**: 녹음 내 이벤트 시퀀스 패턴 파악 → 토크나이징 전략 수립

```
수행 작업:
1. 이벤트 히스토리 JSON 파싱
2. 녹음당 이벤트 수, 이벤트 유형별 빈도 집계
3. Eyes-Open / Eyes-Closed 구간 길이 분포 (class별)
4. Photic stimulation 구간 존재 비율 (모든 녹음에 있는지?)
5. 이벤트 시퀀스 패턴 유형화
   - 전형적 시퀀스: Resting(EO/EC 반복) → Photic Stimulation → 종료
   - 비전형적 시퀀스: 이벤트 누락, 순서 변경 등
6. 이벤트 간 전환 빈도 매트릭스 (어떤 이벤트 다음에 어떤 이벤트가 오는지)
```

**결과물**:
- 이벤트 시퀀스 전형적 패턴 도식
- 이벤트 타입별 구간 길이 통계
- 이벤트 기반 토크나이징에 사용할 구간 정의

### 3.3 주파수 대역별 파워 분석

**목표**: CEEDNet의 occlusion sensitivity 결과를 PSD 분석으로 직접 확인

```
주파수 대역 정의:
- Delta: 0.5–4 Hz
- Theta: 4–8 Hz
- Alpha: 8–13 Hz
- Beta: 13–30 Hz
- Gamma: 30–45 Hz

수행 작업 (Welch's method PSD):
1. 전체 녹음 PSD → Normal vs MCI vs Dementia
2. Eyes-Open 구간 PSD vs Eyes-Closed 구간 PSD (class별)
3. 채널별 PSD (19채널 × 3 class × 2 이벤트)
4. 뇌 영역별 평균 PSD (5영역 × 3 class × 2 이벤트)
5. Alpha/Theta 비율 (class별, 이벤트별)
   - 알려진 바이오마커: 치매 환자에서 Alpha 감소, Theta 증가
6. 통계 검정: Kruskal-Wallis + 사후 Dunn test
```

**핵심 가설 검증**:
- Dementia: Delta/Theta ↑, Alpha/Beta ↓ (기존 신경과학 문헌과 일치 확인)
- MCI: Normal과 Dementia 사이의 중간 패턴 확인
- **MCI 내부의 heterogeneity** 확인 (일부는 Normal에 가깝고, 일부는 Dementia에 가까운지)

### 3.4 Eyes-Open ↔ Eyes-Closed 전환 동역학 분석 ⭐

**목표**: 핵심 가설 — 이벤트 전환 시의 동적 패턴이 class를 구분하는 핵심 특징인지 확인

```
수행 작업:
1. EO→EC 전환 시점 전후 ±5초 구간 추출
2. 전환 전후 Alpha 파워의 시간적 변화 곡선 (class별)
   - 정상: 눈 감으면 Alpha 급격 증가 (Alpha reactivity)
   - 치매: Alpha reactivity 둔화 또는 소실
   - MCI: 중간? 아니면 다른 패턴?
3. 전환 응답 시간 (Alpha가 최대값에 도달하는 데 걸리는 시간)
4. 전환 전후 Theta/Delta 변화 곡선
5. 전환 시 채널 간 동기화 변화 (Phase Locking Value 등)
6. EC→EO 전환에 대해서도 동일 분석

Photic Stimulation 분석:
7. Photic driving response (PDR) 측정
   - 자극 주파수에서의 occipital 영역 파워 증가
   - class별 PDR 강도 비교
8. 자극 주파수별 (3–30Hz) PDR 프로파일 (class별)
```

**이 분석의 중요성**: 만약 전환 동역학에서 Normal/MCI/Dementia 간 유의미한 차이가 관찰되면, 이는 정적 분석으로는 불가능한 **동적 바이오마커**의 존재를 시사하며, EDCC 모델의 설계 근거를 직접 뒷받침합니다.

### 3.5 채널 간 연결성 분석

**목표**: 뇌 네트워크의 class별 차이 확인 → Stage 3 GCN 설계 근거

```
수행 작업:
1. 채널 간 상관관계 행렬 (19×19) — class별
2. 뇌 영역 간 평균 연결 강도 (5×5) — class별
3. Eyes-Open vs Eyes-Closed에서의 연결성 변화 (class별)
4. 전환 시점 전후의 연결성 시간적 변화
5. Graph metrics: clustering coefficient, path length, modularity (class별)
6. 통계 검정: 연결 강도의 class 간 차이
```

**예상 결과**:
- Dementia: 전체적 연결성 감소, 특히 temporal-parietal 연결
- Normal: 영역 간 구조화된 연결 패턴
- MCI: 연결 패턴의 불안정성 (이벤트에 따라 변동이 큰지?)

---

## 4. Phase 2: Baseline 실험

### 4.1 CEEDNet 재현

**목표**: 원 논문의 결과를 재현하여 baseline 성능 확립

```
구현 사항:
- 1D-ResNet-18 (가장 효율적인 단일 모델)
- Random crop augmentation (T=2000, 10초)
- Age signal (age-fc 방식)
- Gaussian noise augmentation
- MixUp regularization
- Test-Time Augmentation (M=8)

평가 메트릭:
- Accuracy, class-wise Sensitivity/Specificity
- ROC-AUC (각 class + macro/micro average)
- Confusion matrix
- 100회 반복 평균 ± std

목표 재현 성능 (원논문 Table 4):
- CEEDNet 1D-ResNet-18 + TTA: ~68.75%
- CEEDNet Ensemble: ~74.66%
```

### 4.2 Foundation Model Fine-tuning Baseline

**목표**: 기존 EEG Foundation Model의 CAUEEG에서의 성능 확인

```
실험 대상 (공개 체크포인트 사용):
1. LaBraM-Base (5.8M params)
   - 체크포인트: https://github.com/935963004/LaBraM
   - CAUEEG-Dementia에 fine-tuning
2. BIOT (3.2M params)
   - 체크포인트: https://github.com/ycq091044/BIOT

Fine-tuning 설정:
- Optimizer: AdamW (lr=5e-4, weight_decay=0.05)
- Scheduler: Cosine annealing
- Epochs: 50 (early stopping patience=10)
- Batch size: 128
- CAUEEG 전처리: 0.5-75Hz bandpass, 200Hz (이미 CAUEEG 기본)

비교 포인트:
- Full fine-tuning vs Linear probing
- "Worth It?" 논문의 주장 검증: linear probing이 정말 약한지?
```

---

## 5. Phase 3: EDCC 모델 구현

### 5.1 전체 아키텍처 개요

```
입력: Raw EEG (19ch × T × 200Hz) + Event history + Age

┌─────────────────────────────────────────────────┐
│  Stage 0: Event-Aware Preprocessing              │
│  - 이벤트 기반 구간 분할                           │
│  - Bandpass filtering + Normalization             │
│  - 이벤트 타입 라벨링                              │
└─────────────┬───────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────┐
│  Stage 1: Cross-scale Tokenization               │
│  - Multi-scale temporal conv (δ,θ,α,β 대응)      │
│  - Regional spatial conv (5 brain regions)        │
│  - Event type embedding + Temporal PE             │
│  출력: x_{c,t} ∈ R^d (per channel, per window)   │
└─────────────┬───────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────┐
│  Stage 2: Dynamic Centralized Core Token          │
│  - CoTAR: channel → core token (per window)       │
│  - Mamba: core token trajectory modeling           │
│  - Event-conditioned state transition             │
│  - Core token redistribution to channels          │
│  출력: h_{c} ∈ R^d (per channel, time-aware)     │
└─────────────┬───────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────┐
│  Stage 3: Region-Level Spatial Reasoning          │
│  - 19ch → 5 region pooling                        │
│  - LapPE + GCN (1-2 layers)                      │
│  출력: z ∈ R^d (graph-level representation)      │
└─────────────┬───────────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────┐
│  Classification Head                              │
│  - Age-conditioned normalization                  │
│  - MLP → 3-class (Normal / MCI / Dementia)       │
│  - Loss: α·L_ce^sam + β·L_ce^sub                 │
└─────────────────────────────────────────────────┘
```

### 5.2 Stage 0: Event-Aware Preprocessing

```python
# 핵심 설계 요소

preprocessing_config = {
    # 기본 신호 처리
    "bandpass": [0.5, 45],         # 60Hz 이상 제거 (CAUEEG는 0.5-70Hz 원본)
    "notch": 50,                    # 한국 전원 주파수
    "sampling_rate": 200,           # CAUEEG 원본 유지
    
    # 이벤트 기반 구간 분할
    "segment_types": [
        "eyes_open",                # EO 구간
        "eyes_closed",              # EC 구간
        "eo_to_ec_transition",      # EO→EC 전환 (전후 ±3초)
        "ec_to_eo_transition",      # EC→EO 전환 (전후 ±3초)
        "photic_stimulation",       # Photic 자극 구간
    ],
    
    # 시간 윈도우
    "window_size": 2.0,             # 2초 (400 samples)
    "window_stride": 1.0,           # 1초 overlap (200 samples)
    
    # 정규화
    "normalization": "z_score",     # 채널별 z-score
    "clip_value": 10,               # ±10 std 이상 클리핑
}
```

### 5.3 Stage 1: Cross-scale Tokenization

```python
# Multi-scale temporal convolution 설계

temporal_conv_config = {
    "kernels": [
        # 커널 1: Beta/Alpha 포착 (12.5-33Hz → 30-80ms 주기)
        {"kernel_size": 15, "out_dim": 64},    # 15/200Hz = 75ms
        
        # 커널 2: Alpha/Theta 포착 (4-13Hz → 77-250ms 주기)
        {"kernel_size": 40, "out_dim": 32},    # 40/200Hz = 200ms
        
        # 커널 3: Delta/Theta 포착 (0.5-8Hz → 125ms-2s 주기)
        {"kernel_size": 100, "out_dim": 16},   # 100/200Hz = 500ms
    ],
    # 최종 토큰 차원: 64 + 32 + 16 = 112 → projection to d=128
    "output_dim": 128,
}

# Spatial tokenization: 5개 뇌 영역 내 multi-scale conv
spatial_conv_config = {
    "brain_regions": {
        "frontal":  ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "FZ"],  # 7ch
        "central":  ["C3", "C4", "CZ"],                              # 3ch
        "parietal": ["P3", "P4", "PZ"],                               # 3ch
        "temporal": ["T3", "T4", "T5", "T6"],                         # 4ch
        "occipital":["O1", "O2"],                                      # 2ch
    },
    "kernels": [
        {"kernel_size": 1, "out_dim": 64},   # 단일 채널 특징
        {"kernel_size": 3, "out_dim": 32},   # 인접 채널 패턴 (영역이 큰 경우)
    ],
    "output_dim": 128,
}

# Embeddings
embedding_config = {
    "temporal_pe": "sinusoidal",     # 시간 위치
    "event_type_embed": "learnable", # 이벤트 타입 (5종)
    "embed_dim": 128,
}
```

### 5.4 Stage 2: Dynamic Centralized Core Token

```python
# CoTAR (Core Token Aggregation-Redistribution)

cotar_config = {
    "input_dim": 128,       # d
    "core_dim": 64,         # Dc (core token 차원)
    "hidden_dim": 128,      # MLP hidden
    "activation": "gelu",
    "num_channels": 19,     # CAUEEG 고정
}

# Mamba for core token trajectory

mamba_config = {
    "d_model": 64,          # core token 차원
    "d_state": 16,          # SSM state 차원
    "d_conv": 4,            # local conv 크기
    "expand": 2,            # expansion factor
    "num_layers": 4,        # Mamba 레이어 수
    
    # Event-conditioned gating
    "event_conditioning": True,
    "event_embed_dim": 16,  # 이벤트 임베딩 → Δ 파라미터 조절
}
```

**Event-Conditioned Mamba의 핵심 메커니즘**:
```
일반 Mamba:
  Δ_t = softplus(f_Δ(x_t))

Event-Conditioned Mamba:
  e_t = EventEmbed(event_type_t)           # 이벤트 타입 임베딩
  Δ_t = softplus(f_Δ([x_t; e_t]))         # 이벤트 정보 반영

  만약 event_t가 "전환"이면:
    → Δ_t가 커짐 → selective forgetting 증가
    → 이전 상태를 많이 잊고, 새 구간 정보를 강하게 수용
  만약 event_t가 "정적 구간"이면:
    → Δ_t가 작음 → 이전 상태 유지
    → 같은 이벤트 내의 점진적 변화 추적
```

### 5.5 Stage 3: Region-Level GCN

```python
gcn_config = {
    "num_nodes": 5,               # 5개 뇌 영역
    "num_gcn_layers": 2,
    "hidden_dim": 128,
    "lapPE_dim": 4,               # Laplacian PE (5노드이므로 최대 4)
    "dropout": 0.3,
    "pooling": "mean",            # 노드 평균 → 그래프 표현
}
```

### 5.6 Classification Head & Loss

```python
classifier_config = {
    # Age-conditioned normalization
    "age_conditioning": {
        "method": "conditional_layer_norm",  # 나이에 따라 정규화 파라미터 변경
        "age_embed_dim": 16,
        "num_age_bins": 10,                  # 10세 단위 binning (50-60, 60-70, ...)
    },
    
    # Optional: Age gradient reversal
    "age_adversarial": {
        "enabled": True,
        "lambda_rev": 0.1,                   # gradient reversal 강도
    },
    
    # Classification MLP
    "hidden_dims": [128, 64],
    "num_classes": 3,
    "dropout": 0.3,
    
    # Loss
    "loss": {
        "sample_ce_weight": 0.5,    # α
        "subject_ce_weight": 0.5,   # β (LEAD식)
        "label_smoothing": 0.1,
    },
}
```

### 5.7 학습 설정

```python
training_config = {
    "optimizer": "AdamW",
    "lr": 1e-4,
    "weight_decay": 0.05,
    "betas": (0.9, 0.999),
    "scheduler": "cosine_annealing",
    "warmup_epochs": 5,
    "total_epochs": 100,
    "early_stopping_patience": 15,
    "batch_size": 64,
    
    # Data augmentation
    "augmentation": {
        "gaussian_noise_std": 0.05,
        "mixup_alpha": 0.2,
        "random_channel_dropout": 0.1,  # 10% 채널 랜덤 마스킹
    },
    
    # LEAD식 Subject-regularized training
    "subject_training": {
        "index_group_shuffling": True,
        "group_size": 8,
    },
    
    # Evaluation
    "eval_repeats": 20,                  # 20회 반복 평균
    "tta_crops": 8,                      # Test-Time Augmentation
    
    # Hardware
    "device": "cuda",
    "num_workers": 4,
    "mixed_precision": True,             # AMP
}
```

---

## 6. Phase 4: Ablation Study 설계

### 6.1 Component Ablation

각 컴포넌트를 하나씩 제거하여 기여도 측정:

| 실험 ID | 설명 | 제거 대상 |
|---------|------|----------|
| A0 | Full EDCC Model | - |
| A1 | No Event-Aware Tokenization | 이벤트 무시, 균일 윈도우 분할 |
| A2 | No Multi-scale Conv | 단일 스케일 커널 (K=1) |
| A3 | No Core Token (CoTAR → Attention) | CoTAR를 standard attention으로 교체 |
| A4 | No Dynamic Core Token | Core token trajectory (Mamba) 제거, static core |
| A5 | No Event Conditioning | Mamba에서 이벤트 정보 제거 |
| A6 | No GCN | Stage 3 제거, Stage 2 출력 직접 분류 |
| A7 | No Age Conditioning | 나이 정보 완전 제거 |
| A8 | No Subject-Level Loss | L_ce^sub 제거, L_ce^sam만 사용 |
| A9 | No Age Adversarial | Gradient reversal 제거 |

### 6.2 이벤트 구간 비교 실험

이벤트 전환 동역학의 진단적 가치를 검증:

| 실험 ID | 사용 구간 | 가설 |
|---------|----------|------|
| E1 | Eyes-Closed 구간만 | 전통적 resting-state 분석 |
| E2 | Eyes-Open 구간만 | EO 상태의 패턴 |
| E3 | EO↔EC 전환 구간만 | 전환 동역학의 단독 가치 |
| E4 | EC + EO (전환 제외) | 정적 구간들만 결합 |
| E5 | EC + EO + 전환 구간 | 정적 + 동적 결합 |
| E6 | 전체 (Photic 포함) | Full pipeline |

**핵심 비교**: E3 vs E4 → 전환 구간만이 정적 구간보다 나은지?  
**핵심 비교**: E5 vs E4 → 전환 구간 추가가 성능을 높이는지?  
**핵심 비교**: E6 vs E5 → Photic stimulation이 추가적 가치를 주는지?

### 6.3 Scaling 실험

| 실험 ID | 변수 | 값 |
|---------|------|-----|
| S1 | 모델 크기 | Small (1M) / Base (5M) / Large (15M) |
| S2 | 학습 데이터 양 | 25% / 50% / 75% / 100% |
| S3 | Mamba 레이어 수 | 1 / 2 / 4 / 8 |
| S4 | Multi-scale 커널 수 | K=1 / K=2 / K=3 |

---

## 7. 평가 메트릭 및 프로토콜

### 7.1 메트릭

```
Primary metrics:
- 3-class Accuracy
- Macro F1-Score
- Class-wise Sensitivity (Recall)
- Class-wise Specificity
- Macro AUROC (one-vs-rest)

Secondary metrics:
- Confusion Matrix (normalized)
- Per-class AUPRC
- Cohen's Kappa
- MCI-specific metrics:
  - MCI Sensitivity (가장 중요: MCI를 놓치지 않는 것)
  - Normal-MCI 구분 AUROC
  - MCI-Dementia 구분 AUROC
```

### 7.2 평가 프로토콜

```
1. CEEDNet 원 논문의 train/val/test split 사용 (공정 비교)
2. 20회 반복 실험 (서로 다른 random seed)
3. 평균 ± 표준편차 보고
4. Test-Time Augmentation (8 crops)
5. 통계 검정: Wilcoxon signed-rank test (baseline과의 유의차)
```

### 7.3 No-Overlap 평가

CEEDNet 원 논문과 동일하게, 환자 중복이 제거된 no-overlap 테스트셋에서도 평가:

```
목적: 환자 중복에 의한 성능 과추정 여부 확인
방법: 원 논문에서 제공하는 no-overlap annotation 사용
보고: Full test vs No-overlap test 성능 비교
```

---

## 8. 시각화 및 해석 계획

### 8.1 Core Token Trajectory 시각화

```
1. t-SNE/UMAP으로 core token trajectory 시각화
   - 시간 축을 따라 core token이 어떻게 이동하는지
   - Normal / MCI / Dementia 별로 색상 구분
   - 이벤트 전환 시점 표시

2. Core token 진화 곡선
   - Core token의 L2 norm 또는 특정 차원을 시간에 따라 플롯
   - 이벤트 전환 시점에서의 급격한 변화 여부

3. Core token에서의 Alpha reactivity 인코딩 확인
   - EC→EO 전환 시 core token 변화량 vs Alpha power 변화량의 상관
```

### 8.2 Brain Region Connectivity 시각화

```
1. 5×5 연결 강도 히트맵 (class별, 이벤트별)
2. GCN에서 학습된 edge weight 시각화
3. Topographic map (MNE 사용)
   - Grad-CAM으로 채널 기여도 시각화
   - class별 뇌 활성화 패턴
```

### 8.3 MCI 분석 특화 시각화

```
1. MCI 오분류 사례 분석
   - Normal로 잘못 분류된 MCI vs Dementia로 잘못 분류된 MCI
   - 이들의 core token trajectory 차이
   - 이들의 주파수 특성 차이

2. MCI 내부 하위 그룹 탐색
   - Core token trajectory 기반 MCI 환자 클러스터링
   - 클러스터별 임상적 특성 (나이, MMSE 등) 비교
```

---

## 9. 예상 결과 및 성공 기준

### 9.1 성능 목표

| 메트릭 | CEEDNet Ensemble (baseline) | EDCC 목표 |
|--------|---------------------------|-----------|
| 3-class Accuracy | 74.66% | **78%+** |
| Normal Sensitivity | 96.98% | 95%+ 유지 |
| MCI Sensitivity | ~50% (추정) | **60%+** |
| Dementia Specificity | 93.59% | 93%+ 유지 |
| Macro AUROC | 0.90 | **0.93+** |

### 9.2 핵심 성공 기준

```
1. [필수] MCI Sensitivity에서 CEEDNet 대비 유의미한 개선
2. [필수] 이벤트 전환 구간이 성능에 기여함을 ablation으로 증명
3. [기대] Core token trajectory에서 class별 구분 가능한 패턴 관찰
4. [기대] Linear probing 성능이 기존 Foundation Model보다 높음
5. [보너스] Core token 동역학이 알려진 신경과학적 바이오마커와 일치
```

---

## 10. 실험 일정 (예상)

| Phase | 작업 | 예상 소요 |
|-------|------|----------|
| **Phase 1** | EDA | 2-3일 |
| **Phase 2** | Baseline 재현 | 2-3일 |
| **Phase 3-1** | Stage 1 구현 + 테스트 | 2-3일 |
| **Phase 3-2** | Stage 2 구현 + 테스트 | 3-4일 |
| **Phase 3-3** | Stage 3 + Full 통합 | 2-3일 |
| **Phase 3-4** | 하이퍼파라미터 튜닝 | 3-5일 |
| **Phase 4** | Ablation + 분석 | 3-5일 |
| **총계** | | **~3-4주** |

---

## 11. 코드 구조 (예상)

```
project/
├── configs/
│   ├── dataset.yaml          # CAUEEG 데이터셋 설정
│   ├── model.yaml            # EDCC 모델 설정
│   └── training.yaml         # 학습 설정
├── data/
│   ├── caueeg_loader.py      # EDF 로더 + 이벤트 파서
│   ├── event_segmenter.py    # 이벤트 기반 구간 분할
│   ├── augmentation.py       # 데이터 증강
│   └── dataset.py            # PyTorch Dataset/DataLoader
├── models/
│   ├── tokenizer.py          # Stage 1: Cross-scale Tokenization
│   ├── cotar.py              # CoTAR 모듈
│   ├── mamba_core.py         # Event-conditioned Mamba
│   ├── gcn.py                # Region-level GCN + LapPE
│   ├── classifier.py         # Age-conditioned classifier
│   └── edcc.py               # Full EDCC 모델
├── baselines/
│   ├── ceednet.py            # CEEDNet 재현
│   └── foundation_ft.py      # Foundation Model fine-tuning
├── analysis/
│   ├── eda.py                # EDA 스크립트
│   ├── frequency_analysis.py # 주파수 분석
│   ├── transition_analysis.py # 전환 동역학 분석
│   ├── connectivity.py       # 연결성 분석
│   └── visualization.py      # 시각화
├── train.py                  # 학습 메인 스크립트
├── evaluate.py               # 평가 스크립트
└── ablation.py               # Ablation 실험
```

---

## 12. 참고 논문 및 코드

| 논문 | 코드 | 참고 대상 |
|------|------|----------|
| CEEDNet (Kim et al., 2023) | github.com/ipis-mjkim/caueeg-ceednet | Baseline, CAUEEG 로딩 |
| LaBraM (Jiang et al., 2024) | github.com/935963004/LaBraM | Foundation Model 체크포인트 |
| LEAD (Wang et al., 2025) | github.com/DL4mHealth/LEAD | Subject-regularized training |
| EvoBrain (Kotoge et al., 2025) | 미공개 | Dynamic graph, Mamba |
| CSBrain (Zhou et al., 2025) | github.com/yuchen2199/CSBrain | Cross-scale tokenization |
| TeCh/CoTAR (Yu et al., 2026) | github.com/Levi-Ackman/TeCh | CoTAR 모듈 |
| Mamba (Gu & Dao, 2024) | github.com/state-spaces/mamba | Mamba 구현 |
