# Implementation of DeiT - Data-Efficient Image Transformer

## Overview
This project implements and evaluates **Data-Efficient Image Transformer (DeiT)**, an improved version of Vision Transformer (ViT) that enhances performance using knowledge distillation. The goal is to compare DeiT with ViT and analyze its effectiveness in image classification on smaller datasets.

## Features
- **DeiT vs ViT:** Comparative analysis of performance.
- **Distillation Token:** Uses a ResNet-50 teacher model to guide training.
- **Data Augmentation:** Implements techniques like CutMix, MixUp, Horizontal Flip, and Random Erasing.
- **Performance Metrics:** Evaluates models using Accuracy, AUC, F1 Score, Precision, and Recall.

## Dataset
- **CIFAR-10:** 50,000 training images and 10,000 test images (32x32 resolution).
- Chosen for its well-labeled structure and availability of pre-trained models.

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed. Install dependencies using:
```bash
pip install torch torchvision timm transformers numpy matplotlib
```


## Training the Model
### Default Training
Train DeiT on CIFAR-10 using:
```bash
python train.py --dataset cifar10 --epochs 20 --batch_size 64 --lr 0.001
```

### Using Knowledge Distillation
To train with a ResNet-50 teacher model:
```bash
python train.py --dataset cifar10 --distillation --teacher_model resnet50
```

## Evaluation
Evaluate the trained model:
```bash
python evaluate.py --model deit --dataset cifar10
```

## Results and Analysis
- **ViT vs DeiT:** DeiT outperforms vanilla ViT on CIFAR-10.
- **Distillation Boost:** Adding a teacher model improves F1 Score and AUC.
- **Data Augmentation:** Enhances accuracy and reduces overfitting.

## Challenges and Solutions
- **High validation loss:** Addressed using data augmentation.
- **Slow CutMix and MixUp:** Reduced probability of application to optimize computation time.

## Future Enhancements
- Extend to object detection tasks.
- Implement DeiT using TensorFlow/Keras.
- Improve interpretability with attention visualizations.

## Authors
- **Vignesh Ram Ramesh Kutti**
- **Aravind Balaji Srinivasan**  


## References
1. Touvron, H., et al. (2020). "Training data-efficient image transformers & distillation through attention." [arXiv](https://doi.org/10.48550/arxiv.2012.12877)
2. Dosovitskiy, A., et al. (2020). "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." [arXiv](https://doi.org/10.48550/arxiv.2010.11929)
3. Vaswani, A., et al. (2017). "Attention is all you need." [arXiv](https://doi.org/10.48550/arxiv.1706.03762)

