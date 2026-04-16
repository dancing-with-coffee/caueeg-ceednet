# CLAUDE.md — Project Guide for Claude Code

## Project Overview

CAUEEG 데이터셋에서 EEG 기반 치매 조기 탐지를 위한 연구 프로젝트.
- **CeedNet**: 기존 baseline 모델 (datasets/, models/, train/, config/)
- **EDCC**: 새로운 Event-Aware Dynamic Centralized Cross-scale 모델 (edcc/)

## Repository Structure

```
caueeg-ceednet/
├── datasets/, models/, train/, config/, optim/  # CeedNet 원본 (수정 금지)
├── edcc/                                         # EDCC 모델 코드
│   ├── data/          # 데이터 파이프라인 (event_segmenter, windowed_dataset, collate, augmentation)
│   ├── models/        # 모델 아키텍처 (tokenizer, cotar, mamba_core, gcn, classifier, edcc_model)
│   ├── training/      # 학습 (losses, trainer)
│   ├── configs/       # 실험 설정 YAML 파일들
│   └── scripts/       # 실행 스크립트
├── notebooks/         # Phase 1 EDA 노트북 (5개)
├── EXPERIMENT_REPORT.md  # 실험 보고서
└── requirements_edcc.txt # EDCC 추가 의존성
```

## How to Run

```bash
# EDCC 학습
python -m edcc.scripts.run_edcc_train --config edcc/configs/edcc_best_combined.yaml

# 다른 seed로 반복
python -m edcc.scripts.run_edcc_train --config edcc/configs/edcc_best_combined.yaml --seed 123

# GPU 선택
CUDA_VISIBLE_DEVICES=1 python -m edcc.scripts.run_edcc_train --config edcc/configs/edcc_large.yaml
```

## Key Conventions

- **CeedNet 코드는 수정하지 않음** — datasets/, models/, train/ 등은 원본 유지
- **EDCC 코드는 edcc/ 내에서만 작업** — 데이터 로딩은 CeedNet의 `datasets.caueeg_script`을 재사용
- **YAML config로 실험 관리** — edcc/configs/ 아래에 실험별 config 파일
- **데이터 경로**: `local/dataset/caueeg-dataset/` (EDF 파일, annotation.json, event JSON)
- **체크포인트 경로**: `local/edcc_checkpoints_*/best_model.pt`

## Data

- CAUEEG-Dementia: Train 950 / Val 119 / Test 118
- 19 EEG channels (10-20 system), 200 Hz, avg 13.3 min
- Event history: Eyes Open/Closed, Photic Stimulation, Artifacts

## Current Best Results

| Config | Test Acc | Balanced Acc | MCI Sens | Dem Sens |
|--------|:--------:|:------------:|:--------:|:--------:|
| Run 5 (edcc_large.yaml) | **61.02%** | - | **53.66%** | 29.03% |
| Run 13+TTA (edcc_best_combined.yaml) | 59.32% | **59.94%** | 31.71% | **74.19%** |

## Environment

- Python 3.13, PyTorch 2.6.0+cu124
- CUDA 12.8, RTX 3090 (24GB)
- mamba-ssm은 ABI 불일치로 native 미사용 → pure-torch SSD 백엔드 사용
