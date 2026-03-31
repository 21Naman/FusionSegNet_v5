# FusionSegNet_v5
# FusionSegNet v5 — Real-Time Drivable Space Segmentation

A from-scratch semantic segmentation model for autonomous vehicle drivable area detection, built for the **Real-Time Drivable Space Segmentation** hackathon track (Problem Statement 2).

> **Restriction compliant:** All weights are initialised randomly and trained from scratch. No pretrained models are used anywhere in the pipeline (`weights=None` throughout).

---

## Overview

FusionSegNet is a multi-sensor fusion segmentation network that combines HD Map, LiDAR ground-plane estimation, Inverse Perspective Mapping (IPM), and lane detection to generate high-confidence pseudo-labels for training. The model outputs a per-pixel classification mask distinguishing drivable space from non-drivable regions (curbs, sidewalks, construction barriers, etc.).

**Dataset:** [nuScenes mini](https://www.nuscenes.org/nuscenes) — all 6 cameras used (CAM_FRONT, CAM_BACK, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK_RIGHT)

---

## Architecture

```
Input Image (3 x H x W)
        |
EfficientNet-B0 Encoder (trained from scratch)
  [stride 2, 4, 8, 16, 32 feature maps]
        |
ASPP Bottleneck (multi-scale context)
  [dilated convolutions: 1, 6, 12, 18]
        |
U-Net Decoder with Attention Gates + SE Blocks
  [skip connections from encoder at each scale]
        |
Deep Supervision Auxiliary Head (at 1/8 resolution)
        |
Output Mask (N_CLASSES x H x W)
```

**Components:**
- **Encoder:** EfficientNet-B0 with compound scaling, initialised from scratch
- **Bottleneck:** Atrous Spatial Pyramid Pooling (ASPP) for multi-scale road features
- **Decoder:** U-Net style with Attention Gates on every skip connection
- **SE Blocks:** Squeeze-and-Excitation channel attention in every decoder block
- **Auxiliary Head:** Deep supervision at 1/8 resolution for faster convergence

---

## Novel Contributions

### 1. Confidence-Weighted Loss (Clever Idea 1)

Rather than treating all pseudo-label pixels equally, each pixel receives a confidence score based on how many sensor modalities agreed on its label:

| Sensor Votes | Confidence |
|---|---|
| 1 (HD Map or LiDAR only) | 0.25 |
| 2 sensors agreed | 0.50 |
| 3 sensors agreed | 0.75 |
| 4 sensors agreed | 1.00 |

During training, the cross-entropy loss for each pixel is multiplied by its confidence score. High-agreement pixels contribute stronger gradient signal. This is equivalent to the model prioritising clean labels without requiring any human annotation to identify which labels are clean.

### 2. Temporal Consistency Self-Supervision (Clever Idea 2)

nuScenes is a video dataset — consecutive frames share the same static road geometry. A KL-divergence penalty between the model's predictions on adjacent frames (for static background regions) provides free self-supervision with zero extra labels. The temporal loss is ramped in gradually after epoch 10 to avoid early training instability.

### 3. 5-View Test-Time Augmentation (Clever Idea 3)

At inference, predictions from 5 augmented views are averaged:
- Original image
- Horizontal flip
- Brightness increase
- Brightness decrease
- Contrast boost

This yields a measurable mIoU improvement at zero additional training cost.

---

## Label Generation Pipeline

The `FusionSegmenter` class fuses 4 sensor modalities to generate pseudo-labels:

1. **HD Map (weight ×2):** Projects nuScenes map layers (drivable surface, walkway, etc.) into the camera image plane using calibrated extrinsics
2. **LiDAR Ground Plane (weight ×2):** Fits a RANSAC plane to LiDAR point cloud, projects ground points into image
3. **Inverse Perspective Mapping (weight ×1):** Detects road surface via IPM and colour-based segmentation in bird's-eye view
4. **Lane Detection (weight ×1):** Hough transform-based lane finding; fills the region between detected lanes

Final labels are determined by majority vote. Confidence scores reflect the degree of agreement across all votes.

---

## Loss Function

```
Total Loss = 0.4 × Confidence-Weighted Cross-Entropy
           + 0.4 × Dice Loss
           + 0.2 × Focal Loss
           + 0.3 × Deep Supervision Loss (auxiliary head)
           + λ   × Temporal Consistency Loss (ramped in after epoch 10)
```

- **Confidence-Weighted CE:** Per-pixel loss scaled by fusion confidence
- **Dice Loss:** Handles class imbalance between road and non-road pixels
- **Focal Loss:** Focuses training on hard examples (road edges, rare classes)
- **Deep Supervision:** Auxiliary CE loss at 1/8 resolution
- **Temporal Consistency:** KL divergence between adjacent frame predictions on static regions

---

## Augmentation Pipeline

Training augmentations (via Albumentations):

- Horizontal flip (p=0.5)
- Random brightness and contrast
- Gaussian noise
- Random rain simulation
- Random fog simulation
- Random shadow
- Optical distortion
- Copy-Paste augmentation (vehicles and pedestrians)

---

## Training

| Parameter | Value |
|---|---|
| Epochs | 45 |
| Batch size | 8 |
| Optimizer | AdamW |
| Scheduler | OneCycleLR |
| Input resolution | 256 × 512 |
| Classes | 5 (background, road, sidewalk, construction, vegetation) |
| All 6 cameras | Yes |

```bash
# Run all cells in the notebook sequentially.
# Cells 1-5:  Install deps, mount drive, extract dataset
# Cell 7:     FusionSegmenter pseudo-label generator
# Cell 9:     Sanity check
# Cells 11-13: Generate masks + copy-paste crops
# Cell 15:    FusionSegNet model definition
# Cells 17-21: Loss functions + dataset + temporal consistency
# Cell 23:    Training loop (run this to train)
# Cells 25-29: Evaluation, TTA, FPS benchmark
# Cells 31-35: Export, visualisation, summary
```

---

## Evaluation Metrics

- **mIoU** — mean Intersection over Union across all classes (primary metric)
- **Binary Drivable IoU** — IoU for the drivable vs. non-drivable binary task
- **FPS** — inference speed at 256×512 resolution on T4 GPU

---

## Edge Case Handling

- **Water puddles:** The IPM module detects high-value, low-saturation regions in bird's-eye view (signature of wet road reflections)
- **Road-to-grass transitions:** LiDAR ground-plane estimation provides geometry-based boundary detection independent of colour
- **Construction barriers:** HD Map layers include `fixed_width_obstacle` annotations fused into the label
- **Adverse weather:** RandomRain and RandomFog augmentations build robustness at training time

---

## Deployment

The model can be exported to TorchScript for inference:

```python
export_torchscript('best_fusionsegnet_v5.pth')
# Outputs: best_fusionsegnet_v5_scripted.pt
```

---

## Project Structure

```
FusionSegNet/
├── FusionSegNet_v5.ipynb    # Main notebook (end-to-end pipeline)
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── .gitignore               # Ignores weights, data, outputs
```

---

## Requirements

See `requirements.txt`. Key dependencies:

- Python 3.9+
- PyTorch 2.0+
- torchvision
- efficientnet-pytorch
- nuscenes-devkit
- albumentations
- opencv-python
- shapely
- pyquaternion

---

## Compliance Note

This project was built under the constraint that **pre-trained models are strictly prohibited**. The EfficientNet-B0 architecture is used as a structural template only. Every weight is randomly initialised and trained from scratch on the nuScenes dataset. This is enforced by passing `weights=None` in every model instantiation.

---

## Dataset

nuScenes mini dataset. Download from the [nuScenes website](https://www.nuscenes.org/nuscenes) and place the `v1.0-mini.tgz` file in your Google Drive root before running the notebook.
