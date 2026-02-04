# Method Details

This document provides a detailed description of the CASA-RCNN architecture and its components.

## Table of Contents

- [1. Overall Framework](#1-overall-framework)
- [2. ConvSwinMerge: Shallow Context Enhancement](#2-convswinmerge-shallow-context-enhancement)
- [3. MambaBlock: Deep Context Modeling](#3-mambablock-deep-context-modeling)
- [4. Quality-Scale Collaborative Loss](#4-quality-scale-collaborative-loss)

---

## 1. Overall Framework

### Problem Definition

Given an aerial image $I \in \mathbb{R}^{3 \times H_0 \times W_0}$, the corresponding set of ground-truth annotations is denoted as $\mathcal{G} = \{(b_i, c_i)\}_{i=1}^{N}$, where $b_i = (x_i, y_i, w_i, h_i)$ represents an axis-aligned bounding box and $c_i$ is the category label. The model outputs a set of detections $\mathcal{D} = \{(\hat{b}_j, \hat{s}_j, \hat{c}_j)\}_{j=1}^{M}$, where $\hat{b}_j$ denotes the predicted bounding box, $\hat{s}_j$ the confidence score, and $\hat{c}_j$ the predicted category.

### Architecture Overview

CASA-RCNN builds upon the standard two-stage detection paradigm with targeted enhancements:

```
Input Image
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backbone (ResNet-50)                      │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐     │
│  │ Stage 0 │──▶│ Stage 1 │──▶│ Stage 2 │──▶│ Stage 3 │     │
│  │   C1    │   │   C2    │   │   C3    │   │   C4    │     │
│  └────┬────┘   └────┬────┘   └────┬────┘   └─────────┘     │
│       │             │             │                          │
│       ▼             ▼             ▼                          │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                   │
│  │ConvSwin │   │ConvSwin │   │ Mamba   │                   │
│  │  Merge  │   │  Merge  │   │  Block  │                   │
│  └────┬────┘   └────┬────┘   └────┬────┘                   │
└───────┼─────────────┼─────────────┼─────────────────────────┘
        │             │             │
        ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Feature Pyramid Network                   │
│              P2 ◀── P3 ◀── P4 ◀── P5 ◀── P6                │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────┐     ┌─────────────────────────────────────┐
│       RPN       │────▶│              RoI Head               │
│   (Proposals)   │     │  ┌───────────┐  ┌────────────────┐ │
└─────────────────┘     │  │ Varifocal │  │ScaleAdaptive   │ │
                        │  │   Loss    │  │     Loss       │ │
                        │  └───────────┘  └────────────────┘ │
                        └─────────────────────────────────────┘
```

### Module Placement

| Module | Location | Channel Dimension | Purpose |
|--------|----------|-------------------|---------|
| ConvSwinMerge | Stage 0, Stage 1 | 256, 512 | Shallow context enhancement for small objects |
| MambaBlock | Stage 2 | 1024 | Deep global dependency modeling |

---

## 2. ConvSwinMerge: Shallow Context Enhancement

### Design Motivation

In UAV aerial imagery, objects often occupy only a few pixels with high-density distributions. Shallow high-resolution features are indispensable for small-object detection because:

1. **Preserving Details**: Shallow features retain rich edges, corners, and texture information
2. **Background Interference**: High-frequency responses may strongly activate background structures (road markings, building outlines, shadows)
3. **Dual Requirements**: Need both fine-grained representation and background suppression

### Architecture

ConvSwinMerge adopts a serial residual strategy with three components:

```
Input X ──┬──▶ CoordAtt ──▶ (+) ──▶ Conv3x3 ──▶ (+) ──▶ SaE ──▶ Output Y
          │                 ▲                    ▲
          └─────────────────┘                    │
                            └────────────────────┘
```

### Forward Propagation

Given shallow input feature $X \in \mathbb{R}^{C \times H \times W}$:

$$X_1 = X + \text{CoordAtt}(X)$$
$$X_2 = X_1 + \text{Conv}_{3\times3}(X_1)$$
$$Y = \text{SaE}(X_2)$$

### Sub-module Details

#### 2.1 Coordinate Attention (CoordAtt)

Injects positional information into channel recalibration by encoding features along horizontal and vertical directions:

- **Channel Selectivity**: Learns which channels are important
- **Spatial Localizability**: Learns where to focus attention
- **Position Sensitivity**: Critical for small aerial targets

#### 2.2 Local Convolution (Conv)

Performs local neighborhood aggregation:

- **Edge Enhancement**: Strengthens local consistency of edges and textures
- **Detail Preservation**: Avoids weakening of local structures from attention-only approaches
- **Kernel Size**: 3×3 convolution for local receptive field

#### 2.3 Squeeze-and-Excitation Layer (SaE)

Channel-wise recalibration through grouped/multi-branch excitation:

- **Channel Subspace Learning**: Different channel subspaces learn finer-grained response combinations
- **Background Discrimination**: Enables effective discrimination between small objects and textured backgrounds

### Impact on Two-Stage Detection

1. **RPN Stage**: More discriminative shallow features improve recall for small-object proposals while reducing spurious proposals from background textures
2. **RoI Refinement**: Clearer boundaries and stable foreground responses help the regression branch learn more accurate geometric offsets

---

## 3. MambaBlock: Deep Context Modeling

### Design Motivation

In UAV aerial imagery, targets exhibit:
- **High-density clustering**
- **Frequent occlusion and overlap**
- **Similar inter-class appearances**

Local convolutional aggregation leads to semantic ambiguity. Deep features need global dependency modeling for robust detection in dense and occluded conditions.

### Architecture

MambaBlock uses a dual-branch fusion design:

```
Input X ──┬──▶ Enhancement Branch M(·) ──┐
          │                               ├──▶ Concat ──▶ φ(·) ──▶ Output Y
          └──▶ Fidelity Branch I(·) ──────┘
```

### Forward Propagation

Given deep input features $X \in \mathbb{R}^{C \times H \times W}$:

$$X_m = \mathcal{M}(X)$$
$$X_i = \mathcal{I}(X)$$
$$Y = \phi(\text{Concat}(X_m, X_i))$$

Where:
- $\mathcal{M}(\cdot)$: Enhancement branch with adaptive sequence modeling
- $\mathcal{I}(\cdot)$: Fidelity branch (identity mapping or lightweight linear transform)
- $\phi(\cdot)$: 1×1 fusion convolution

### Enhancement Branch: MambaT

The enhancement branch treats deep features as a sequence and performs content-aware selection:

**Step 1: Coarse-grained Key Extraction**
$$K_1 = \text{GroupConv}(X)$$

**Step 2: Value Generation**
$$V = \text{Conv}_{1\times1}(X)$$

**Step 3: Spatial Weight Generation**
$$A = \text{Softmax}(\psi([K_1; X]))$$

Where $\psi(\cdot)$ is a lightweight mapping function that generates spatial weights.

**Step 4: Adaptive Aggregation**
$$K_2 = \text{Reshape}(A \odot \text{Flatten}(V))$$

**Step 5: Residual Output**
$$\mathcal{M}(X) = K_1 + K_2$$

### Design Rationale

- **$K_1$ (Local Pedestal)**: Ensures stable local semantic extraction
- **$K_2$ (Adaptive Term)**: Controlled by content-driven weights $A$, enables cross-region information integration
- **Dual-branch Fusion**: Context modeling acts as incremental supplement, preventing over-smoothing

---

## 4. Quality-Scale Collaborative Loss

### Overview

The training objective extends beyond "category classification + bounding-box regression" to:
1. **Align classification scores with localization quality**
2. **Adaptively reweight regression across scales**

### 4.1 Varifocal Loss

For the classification branch of the RoI head:

$$\mathcal{L}_{\text{VFL}}(p, q) = \begin{cases} -q|q-p|^\gamma \log(p), & q > 0 \\ -\alpha p^\gamma \log(1-p), & q = 0 \end{cases}$$

Where:
- $p \in (0, 1)$: Predicted probability (Sigmoid output)
- $q$: Quality target
  - For positive samples: $q = \text{IoU}(\hat{b}, b) \in (0, 1]$
  - For negative samples: $q = 0$
- $\gamma = 2.0$: Focusing parameter
- $\alpha = 0.75$: Negative sample weight

**Benefits**:
- Amplifies contribution of high-IoU positive samples
- Encourages consistency between classification scores and localization quality
- Prioritizes high-quality proposals in crowded scenes

### 4.2 ScaleAdaptiveLoss

For the regression branch, addressing the issue that small objects have smaller pixel-space localization errors and thus contribute less gradient:

$$\mathcal{L}_{\text{scale}} = w(s) \cdot \mathcal{L}_{\text{reg}}^{\text{base}}(\hat{b}, b)$$

Scale measure: $s = \sqrt{wh}$

Scale-dependent weights:

$$w(s) = \begin{cases} \lambda_s = 3.0, & 0 \leq s < 32 \text{ (small)} \\ \lambda_m = 2.0, & 32 \leq s < 96 \text{ (medium)} \\ \lambda_l = 1.0, & s \geq 96 \text{ (large)} \end{cases}$$

### 4.3 Total Loss

The RoI head objective:

$$\mathcal{L}_{\text{head}} = \lambda_{\text{cls}} \mathcal{L}_{\text{VFL}} + \lambda_{\text{reg}} \mathcal{L}_{\text{scale}}$$

The overall optimization objective:

$$\mathcal{L} = \mathcal{L}_{\text{rpn-cls}} + \mathcal{L}_{\text{rpn-reg}} + \lambda_{\text{cls}} \mathcal{L}_{\text{VFL}} + \lambda_{\text{reg}} \mathcal{L}_{\text{scale}}$$

### Loss Configuration

| Stage | Loss Type | Implementation | Weight |
|-------|-----------|----------------|--------|
| RPN | Classification | Focal Loss ($\gamma=2.0$, $\alpha=0.25$) | 1.0 |
| RPN | Regression | GIoU Loss | 2.0 |
| RoI Head | Classification | Varifocal Loss ($\gamma=2.0$, $\alpha=0.75$) | 1.0 |
| RoI Head | Primary Regression | EIoU Loss | 2.5 |
| RoI Head | Auxiliary Regression | ScaleAdaptiveLoss | 1.5 |

---

## Summary

CASA-RCNN addresses the unique challenges of UAV aerial object detection through:

1. **Hierarchical Feature Enhancement**: ConvSwinMerge for shallow features, MambaBlock for deep features
2. **Quality-Scale Collaborative Training**: Varifocal Loss for ranking, ScaleAdaptiveLoss for small-object localization
3. **Preserved Inference Pipeline**: All enhancements are compatible with standard two-stage detection inference
