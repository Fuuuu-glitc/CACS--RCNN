# Experimental Results

This document provides detailed experimental results and analysis for CASA-RCNN.

## Table of Contents

- [1. Dataset](#1-dataset)
- [2. Implementation Details](#2-implementation-details)
- [3. Main Results](#3-main-results)
- [4. Per-Category Results](#4-per-category-results)
- [5. Ablation Studies](#5-ablation-studies)
- [6. Efficiency Analysis](#6-efficiency-analysis)

---

## 1. Dataset

### VisDrone2021-DET

VisDrone is a large-scale benchmark dataset for UAV vision tasks released by the Machine Learning and Data Mining Lab at Tianjin University. We use the **VisDrone2021-DET** subset for static image object detection.

#### Dataset Characteristics

- **Diverse Collection Scenes**: Urban streets, residential areas, parks, highways, campuses
- **Varying Conditions**: Day/night, clear/cloudy/foggy weather
- **Flight Heights**: 5m to 300m with tilt angles from vertical to 45°
- **Dense Objects**: 50-200 instances per image on average

#### Dataset Statistics

| Split | Images | Objects | Avg Objects/Image | Resolution |
|-------|--------|---------|-------------------|------------|
| Train | 6,471 | 343,205 | 53.0 | 1920×1080 |
| Val | 548 | 38,759 | 70.7 | 1920×1080 |
| Test-dev | 1,610 | -- | -- | 1920×1080 |

#### Object Scale Distribution

| Scale | Area Range | Proportion |
|-------|------------|------------|
| Small | $a < 32^2$ | ~55% |
| Medium | $32^2 \leq a < 96^2$ | ~35% |
| Large | $a \geq 96^2$ | ~10% |

#### Category Distribution

| ID | Category | Train Samples | Proportion | Typical Scale |
|----|----------|---------------|------------|---------------|
| 1 | pedestrian | 78,654 | 22.9% | Small |
| 2 | people | 32,187 | 9.4% | Small |
| 3 | bicycle | 11,423 | 3.3% | Small |
| 4 | car | 144,865 | 42.2% | Medium |
| 5 | van | 25,678 | 7.5% | Medium |
| 6 | truck | 13,456 | 3.9% | Large |
| 7 | tricycle | 8,932 | 2.6% | Medium |
| 8 | awning-tricycle | 5,123 | 1.5% | Medium |
| 9 | bus | 5,892 | 1.7% | Large |
| 10 | motor | 14,567 | 4.2% | Small |
| 11 | others | 2,428 | 0.7% | Mixed |

---

## 2. Implementation Details

### Network Architecture

| Component | Configuration |
|-----------|--------------|
| **Backbone** | ResNet-50, ImageNet pretrained |
| **Frozen Stages** | Stage 0 (conv1 + bn1) |
| **ConvSwinMerge** | Stage 0 (256 channels), Stage 1 (512 channels) |
| **MambaBlock** | Stage 2 (1024 channels) |
| **FPN Output Channels** | 256 |
| **FPN Output Levels** | 5 ($P_2$ - $P_6$) |

### RPN Configuration

| Parameter | Value |
|-----------|-------|
| Anchor Scales | {8} |
| Anchor Ratios | {0.5, 1.0, 2.0} |
| Anchor Strides | {4, 8, 16, 32, 64} |
| Positive IoU Threshold | 0.7 |
| Negative IoU Threshold | 0.3 |
| Training Samples | 256/image (1:1 pos/neg ratio) |

### RoI Head Configuration

| Parameter | Value |
|-----------|-------|
| RoI Feature Size | 7×7 |
| FC Dimension | 1024 |
| Positive IoU Threshold | 0.5 |
| Training Samples | 512/image (1:3 pos/neg ratio) |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Initial Learning Rate | 2×10⁻³ |
| Weight Decay | 1×10⁻⁴ |
| Gradient Clipping | max_norm = 0.1 |
| Batch Size | 6 (single GPU) |
| Total Iterations | 20,000 |
| Validation Interval | 5,000 iterations |
| LR Schedule | MultiStepLR |
| LR Decay Step | 15,000 iterations |
| LR Decay Factor | γ = 0.1 |
| Backbone LR Multiplier | 0.1 |

### Data Augmentation

| Augmentation | Configuration |
|--------------|---------------|
| Input Size | 1333×800 |
| Random Horizontal Flip | Probability 0.5 |
| Keep Aspect Ratio | No (fixed size) |

### Hardware Environment

- **GPU**: NVIDIA RTX 3090 (24GB)
- **CPU**: Intel Core i9-10900K
- **RAM**: 64GB DDR4
- **Framework**: PyTorch 1.12.0 + CUDA 11.6
- **Detection Framework**: MMDetection 3.0

---

## 3. Main Results

### Overall Performance Comparison

| Method | Type | mAP | mAP@50 | mAP@75 | mAP_s | mAP_m | mAP_l |
|--------|------|-----|--------|--------|-------|-------|-------|
| SSD | One-stage | 3.6 | 8.4 | 2.6 | 0.5 | 5.5 | 12.3 |
| Deformable DETR | Transformer | 7.1 | 15.0 | 6.0 | 3.3 | 11.4 | 15.2 |
| Fast R-CNN | Two-stage | 12.8 | 23.4 | 12.9 | 6.4 | 20.0 | 25.6 |
| DINO | Transformer | 13.0 | 24.7 | 12.5 | 7.7 | 20.2 | 25.5 |
| Faster R-CNN | Two-stage | 13.9 | 24.7 | 14.4 | 6.9 | 21.9 | 23.1 |
| RetinaNet | One-stage | 14.5 | 25.8 | 14.8 | 5.9 | 23.7 | 32.4 |
| DDOD | One-stage | 14.7 | 26.2 | 14.7 | 6.9 | 23.1 | 30.9 |
| **CASA-RCNN** | Two-stage | **22.9** | **36.6** | **25.7** | **12.5** | **35.7** | **37.9** |

### Improvement Analysis

| Metric | Faster R-CNN | CASA-RCNN | Absolute Δ | Relative Gain |
|--------|--------------|-----------|------------|---------------|
| mAP | 13.9% | 22.9% | +9.0% | **+64.7%** |
| mAP@50 | 24.7% | 36.6% | +11.9% | +48.2% |
| mAP@75 | 14.4% | 25.7% | +11.3% | +78.5% |
| mAP_s | 6.9% | 12.5% | +5.6% | **+81.2%** |
| mAP_m | 21.9% | 35.7% | +13.8% | +63.0% |
| mAP_l | 23.1% | 37.9% | +14.8% | +64.1% |

### Key Observations

1. **Overall Performance**: CASA-RCNN achieves 22.9% mAP, significantly outperforming all comparison methods
2. **Small Object Detection**: mAP_s improves from 6.9% to 12.5% (+81.2% relative gain), validating ConvSwinMerge and ScaleAdaptiveLoss effectiveness
3. **Medium/Large Objects**: mAP_m and mAP_l also show substantial improvements, demonstrating MambaBlock's global context modeling capability
4. **Transformer Methods**: Deformable DETR and DINO underperform on this dataset, possibly due to high object density and extreme scale variation

---

## 4. Per-Category Results

### Detailed Per-Category AP (%)

| Method | ped. | peo. | bic. | car | van | truck | tri. | a-tri. | bus | motor | others | mAP |
|--------|------|------|------|-----|-----|-------|------|--------|-----|-------|--------|-----|
| Faster R-CNN | 9.3 | 3.3 | 3.3 | 42.4 | 24.6 | 17.0 | 7.7 | 5.1 | 31.6 | 8.2 | 0.7 | 13.9 |
| RetinaNet | 8.8 | 4.1 | 4.7 | 41.2 | 22.0 | 19.6 | 8.3 | 5.9 | 34.3 | 7.7 | 3.5 | 14.5 |
| DDOD | 10.0 | 3.6 | 4.3 | 43.1 | 23.5 | 17.8 | 6.9 | 6.2 | 34.8 | 9.0 | 2.2 | 14.7 |
| DINO | 10.2 | 6.5 | 3.7 | 40.2 | 19.4 | 13.1 | 7.4 | 6.4 | 24.3 | 11.3 | 1.6 | 13.0 |
| **CASA-RCNN** | **11.5** | **4.6** | **7.2** | **48.1** | **39.9** | **39.7** | **13.8** | **13.2** | **52.0** | **13.4** | **8.7** | **22.9** |

*Note: ped.=pedestrian, peo.=people, bic.=bicycle, tri.=tricycle, a-tri.=awning-tricycle*

### Improvement over Faster R-CNN

| Category | Faster R-CNN | CASA-RCNN | Δ |
|----------|--------------|-----------|---|
| pedestrian | 9.3 | 11.5 | +2.2 |
| people | 3.3 | 4.6 | +1.3 |
| bicycle | 3.3 | 7.2 | **+3.9** |
| car | 42.4 | 48.1 | +5.7 |
| van | 24.6 | 39.9 | **+15.3** |
| truck | 17.0 | 39.7 | **+22.7** |
| tricycle | 7.7 | 13.8 | +6.1 |
| awning-tricycle | 5.1 | 13.2 | +8.1 |
| bus | 31.6 | 52.0 | **+20.4** |
| motor | 8.2 | 13.4 | +5.2 |
| others | 0.7 | 8.7 | **+8.0** |

### Category-wise Analysis

- **Small Object Categories** (pedestrian, people, bicycle, motor): Improvements of 1.3-3.9 points, attributed to shallow-level enhancement and scale-adaptive loss
- **Large Object Categories** (truck, bus): Massive improvements of 20-23 points, demonstrating MambaBlock's semantic understanding capability
- **Difficult Categories** (awning-tricycle, others): Significant improvements on rare and visually diverse categories, showing better generalization

---

## 5. Ablation Studies

### 5.1 Core Module Ablation

| ConvSwinMerge | MambaBlock | Q-S Loss | mAP | mAP@50 | mAP@75 | mAP_s | mAP_m | mAP_l |
|:-------------:|:----------:|:--------:|-----|--------|--------|-------|-------|-------|
| | | | 13.9 | 24.7 | 14.4 | 6.9 | 21.9 | 23.1 |
| ✓ | | | 18.7 | 31.7 | 19.9 | 9.8 | 29.3 | 36.9 |
| | ✓ | | 19.1 | 32.2 | 20.6 | 10.1 | 29.7 | 35.7 |
| ✓ | ✓ | | 22.9 | 36.6 | 25.7 | 12.5 | 35.7 | 37.9 |
| ✓ | ✓ | ✓ | **25.8** | **39.2** | **31.0** | **16.7** | **38.9** | **40.2** |

*Q-S Loss = Quality-Scale Collaborative Loss (Varifocal Loss + ScaleAdaptiveLoss)*

**Key Findings**:
- **ConvSwinMerge alone**: +4.8% mAP, +2.9% mAP_s (validates shallow context enhancement)
- **MambaBlock alone**: +5.2% mAP, +12.6% mAP_l (validates deep global modeling)
- **Combined modules**: 22.9% mAP (synergistic effect: shallow features provide reliable basis for deep modeling)
- **With Q-S Loss**: +2.9% additional mAP, +4.2% mAP_s (quality-aware and scale-adaptive optimization)

### 5.2 ConvSwinMerge Sub-module Ablation

| CoordAtt | Conv | SaE | mAP | mAP_s | mAP_m | ΔmAP |
|:--------:|:----:|:---:|-----|-------|-------|------|
| | | | 13.9 | 6.9 | 21.9 | -- |
| ✓ | | | 15.8 | 8.2 | 24.3 | +1.9 |
| | ✓ | | 14.6 | 7.3 | 22.5 | +0.7 |
| | | ✓ | 14.9 | 7.5 | 23.1 | +1.0 |
| ✓ | ✓ | | 17.1 | 8.9 | 26.8 | +3.2 |
| ✓ | | ✓ | 16.8 | 8.7 | 26.2 | +2.9 |
| ✓ | ✓ | ✓ | **18.7** | **9.8** | **29.3** | **+4.8** |

**Analysis**:
- **CoordAtt** provides the largest individual contribution (+1.9%), validating position-sensitive modeling importance
- **Combined effect** (+4.8%) exceeds sum of individual effects (+3.6%), indicating synergistic enhancement

### 5.3 ScaleAdaptiveLoss Weight Ablation

| λ_s | λ_m | λ_l | mAP | mAP_s | mAP_m | mAP_l |
|-----|-----|-----|-----|-------|-------|-------|
| 1.0 | 1.0 | 1.0 | 22.9 | 12.5 | 35.7 | 37.9 |
| 2.0 | 1.5 | 1.0 | 24.1 | 14.3 | 36.8 | 38.5 |
| **3.0** | **2.0** | **1.0** | **25.8** | **16.7** | **38.9** | **40.2** |
| 4.0 | 2.5 | 1.0 | 25.2 | 17.1 | 37.8 | 38.9 |
| 5.0 | 3.0 | 1.0 | 24.6 | 16.8 | 36.5 | 37.2 |

**Analysis**:
- Optimal weights: **λ_s:λ_m:λ_l = 3:2:1**
- Excessive small-object weights (5:3:1) improve mAP_s but degrade medium/large object performance
- Balance is needed across scales for optimal overall performance

### 5.4 Loss Function Combination Ablation

| Classification Loss | Regression Loss | mAP | mAP@75 | mAP_s | ΔmAP |
|---------------------|-----------------|-----|--------|-------|------|
| CrossEntropy | L1 Loss | 22.9 | 25.7 | 12.5 | -- |
| CrossEntropy | EIoU Loss | 23.5 | 27.2 | 13.1 | +0.6 |
| Varifocal | L1 Loss | 23.8 | 28.1 | 13.5 | +0.9 |
| Varifocal | EIoU Loss | 24.3 | 29.5 | 14.2 | +1.4 |
| **Varifocal** | **EIoU + ScaleAdaptive** | **25.8** | **31.0** | **16.7** | **+2.9** |

**Analysis**:
- **EIoU vs L1**: +0.6% mAP, +1.5% mAP@75 (IoU-based loss improves localization)
- **Varifocal vs CE**: +0.9% mAP, +2.4% mAP@75 (quality-aware classification improves high-IoU detection)
- **ScaleAdaptiveLoss**: Additional +1.5% mAP, +2.5% mAP_s (effective compensation for small object training)

---

## 6. Efficiency Analysis

### Computational Cost Comparison

| Model | Params (M) | FLOPs (G) | FPS | mAP (%) |
|-------|------------|-----------|-----|---------|
| Faster R-CNN | 41.4 | 207.1 | 18.2 | 13.9 |
| RetinaNet | 36.3 | 205.8 | 19.5 | 14.5 |
| Deformable DETR | 40.1 | 173.2 | 12.8 | 7.1 |
| DINO | 47.2 | 268.4 | 8.6 | 13.0 |
| **CASA-RCNN** | 48.6 | 235.8 | 15.4 | **22.9** |

### Overhead Analysis

| Metric | Faster R-CNN | CASA-RCNN | Increment | Ratio |
|--------|--------------|-----------|-----------|-------|
| Parameters | 41.4M | 48.6M | +7.2M | +17.4% |
| FLOPs | 207.1G | 235.8G | +28.7G | +13.9% |
| FPS | 18.2 | 15.4 | -2.8 | -15.4% |
| mAP | 13.9% | 22.9% | +9.0% | **+64.7%** |

### Efficiency-Performance Trade-off

- **Parameter increase**: +17.4% (acceptable overhead)
- **FLOPs increase**: +13.9% (moderate computational cost)
- **Speed decrease**: -15.4% (still faster than Transformer methods)
- **Performance gain**: +64.7% relative mAP improvement

CASA-RCNN maintains **significantly faster inference speed** (15.4 FPS) compared to Transformer-based methods (8.6-12.8 FPS), making it more suitable for practical deployment in UAV applications.

---

## Summary

CASA-RCNN demonstrates:

1. **State-of-the-art performance** on VisDrone2021 with 22.9% mAP
2. **Significant small object improvement** with 81.2% relative gain on mAP_s
3. **Consistent improvements across all categories** and object scales
4. **Efficient design** with acceptable computational overhead
5. **Complementary module contributions** validated through comprehensive ablations
