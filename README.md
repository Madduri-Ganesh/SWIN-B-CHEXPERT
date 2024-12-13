# Swin-B Transformer for Chest X-ray Classification

## Overview
This repository contains the implementation of a **Swin-B (Base) Transformer** model for chest X-ray image classification using the **CheXpert v1.0** dataset. Swin-B is a larger variant of the Swin Transformer architecture, offering improved performance across various computer vision tasks while maintaining efficiency through its shifted window approach.

## Model Architecture
The Swin-B model used in this project has the following characteristics:
- **Embedding Dimensions**: 128 in the first stage, scaling up to 1024 in the last stage
- **Total Parameters**: Approximately 88 million
- **Architecture**: Based on the Swin Transformer, featuring shifted windows for efficient attention computation

## Dataset: CheXpert v1.0
This project utilizes the **CheXpert v1.0** dataset, a large public chest radiograph dataset for medical imaging research and development.

### Key Details
- **Size**: 224,316 chest radiographs from 65,240 patients
- **Image Types**: Frontal and lateral chest X-rays
- **Labels**: 14 common chest radiographic observations
- **Label Types**: Positive (1), Uncertain (-1), Negative (0), or Not Mentioned

### Significance
CheXpert v1.0 has become one of the most widely used and cited clinical AI datasets in radiology. Its large scale, diverse patient population, and inclusion of uncertainty labels make it particularly valuable for developing and evaluating machine learning models for chest X-ray interpretation.

## Training Details
The model was trained with the following parameters and configuration:
- **Pre-training**: Utilized a pre-trained architecture as a starting point
- **Epochs**: 25
- **Initial Learning Rate**: 0.005
- **Loss Function**: BCEWithLogitsLoss
- **Optimizer**: AdamW
- **Data Loading**: 10 worker threads
- **Batch Size**: 32
- **Learning Rate Scheduler**: ReduceLROnPlateau
  - **Factor**: 0.1
  - **Patience**: 3 epochs
