
# Safety-Oriented Evaluation of Pedestrian Detection in Autonomous Driving

## 1. Overview

This project investigates the performance of vision-based pedestrian detection methods in autonomous driving, with a particular focus on safety-critical situations.

The experiments compare individual detection models, methods for improving individual models, and ensemble methods using both conventional computer vision metrics and safety-related metrics.

The main objectives are to:
- Evaluate pedestrian detection performance on image sequences.
- Investigate differences between overall detection performance and safety-related detection performance.
- Analyze detection failures in safety-critical situations.
- Explore methods to improve safety-related detection performance.

## 2. Experimental Design

### 2.1 Datasets

| Dataset | Description |
|---------|-------------|
| nuScenes | Autonomous driving dataset with multi-camera image sequences and 3D annotations |
| KITTI | Autonomous driving dataset with image sequences and object annotations, depending on the selected benchmark |

**Task:** Pedestrian detection in image sequences.

**Input:** Camera images from autonomous driving scenarios.

**Output:** Pedestrian bounding boxes, detection confidence scores, and frame-level detection results.

### 2.2 Methods

The evaluated methods are organized into two categories.

#### Type 1: Individual Model Improvement

Methods designed to improve the detection performance of individual models.

Examples:
- Baseline object detectors
- Model fine-tuning
- Other individual model enhancement methods

#### Type 2: Ensemble Methods

Methods that combine multiple detection models to improve detection performance and reliability.

Examples:
- Voting-based ensembles
- Confidence-based fusion
- Adaptive ensemble methods

### 2.3 Evaluation Metrics

Two categories of metrics are considered.

#### A. Vision Metrics

Conventional metrics for evaluating object detection performance.

- Average Precision (AP)
- Mean Average Precision (mAP)
- Precision
- Recall

#### B. Safety-Related Metrics

Metrics that evaluate detection performance in safety-critical situations.

Candidate metrics include:
- Recall on safety-critical objects/frames
- Missed detections in safety-critical situations
- Consecutive missed detections
- Late initial detection

The final selection and definitions of safety-related metrics will be determined during the experimental study.

### 2.4 Safety-Critical Annotation

Safety-critical situations are identified using the spatial and temporal relationships between the ego vehicle and pedestrians.

Potential criteria include:
- Time-to-Collision (TTC)
- Reachability analysis
- Stopping-distance constraints

Each pedestrian instance can be assigned a frame-level state:

| Label | Description |
|-------|-------------|
| 0 | Pedestrian absent or outside the evaluation scope |
| 1 | Pedestrian present but not safety-critical |
| 2 | Pedestrian present and safety-critical |

These labels support separate evaluations of overall detection performance and safety-critical detection performance.

## 3. Experimental Workflow

1. **Data Preparation**
   - Load image sequences and pedestrian annotations.
   - Identify pedestrian instances across frames.
   - Generate or load safety-critical annotations.

2. **Model Inference**
   - Run individual object detectors.
   - Apply individual model improvement methods.
   - Apply ensemble methods.
   - Store detection results.

3. **Performance Evaluation**
   - Match predictions with ground-truth annotations.
   - Calculate conventional vision metrics.
   - Calculate safety-related metrics.

4. **Comparative Analysis**
   - Compare different individual models.
   - Compare individual model improvements with ensemble methods.
   - Analyze differences between vision and safety-related performance.
   - Investigate temporal patterns of detection failures.

## 4. Experimental Results

Results will be organized to compare performance across datasets, methods, and evaluation metrics.

| Dataset | Method | Type | Vision Metrics | Safety Metrics |
|---------|--------|------|----------------|----------------|
| nuScenes | Baseline Model A | Baseline | TBD | TBD |
| nuScenes | Improved Model A | Type 1 | TBD | TBD |
| nuScenes | Ensemble A | Type 2 | TBD | TBD |
| KITTI | Baseline Model A | Baseline | TBD | TBD |
| KITTI | Improved Model A | Type 1 | TBD | TBD |
| KITTI | Ensemble A | Type 2 | TBD | TBD |


