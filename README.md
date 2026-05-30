# Challenge 7 - Group 3
## Transfer Learning: Few-Shot Classification, Neural Style Transfer, and Domain Shift Adaptation

### Overview
This project implements the complete pipeline required for Challenge 7 of the Machine Learning course.

The objective is to study domain shift between real-world photographs and artistic paintings using the DomainNet dataset. The project is divided into three parts:

1. Few-Shot Classification with Transfer Learning
2. Neural Style Transfer
3. Domain Adaptation

Our assigned task is:

- Source Domain: Real
- Target Domain: Painting
- Categories:
  - Beach
  - Bridge
  - Mountain
  - River
  - Tree

The goal is to quantify the performance degradation caused by domain shift and evaluate different adaptation strategies.

---

## Repository Structure

```text
challenge-7_group3/
│
├── checkpoints/
│   ├── resnet50_feature_extraction.pth
│   └── ...
│
├── data/
│   ├── DomainNet/
│   │   ├── real/
│   │   │   ├── beach/
│   │   │   ├── bridge/
│   │   │   ├── mountain/
│   │   │   ├── river/
│   │   │   └── tree/
│   │   │
│   │   └── painting/
│   │       ├── beach/
│   │       ├── bridge/
│   │       ├── mountain/
│   │       ├── river/
│   │       └── tree/
│   │
│   └── synthetic_target/
│       ├── beach/
│       ├── bridge/
│       ├── mountain/
│       ├── river/
│       └── tree/
│
├── figures/
│
├── notebooks/
│   ├── part_a_few_shot.ipynb
│   ├── part_b_style_transfer.ipynb
│   └── part_c_domain_adaptation.ipynb
│
├── classifier.py
├── style_transfer.py
├── domain_adaptation.py
├── requirements.txt
├── CHECKLIST.md
└── README.md
```
---
# Authors
Group #3
David Buitrago 20221020085 //
Cristian Cruz 202210200125 //
Daniel Cuellar 20221020081


Transfer Learning: Few-Shot Classification, Neural Style Transfer, and Domain Shift Adaptation
