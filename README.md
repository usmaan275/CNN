# 🧠 Custom CNN with Expert Routing for CIFAR-10
A neural network project for the CIFAR-10 dataset, implementing a custom convolutional neural network architecture featuring Expert Blocks with dynamic routing, trained using PyTorch.

## 📂 Project Overview
This project is part of the Neural Networks and Deep Learning coursework and focuses on designing, training, and evaluating a modular CNN that dynamically routes inputs through expert convolutional branches. The model is trained and tested on the CIFAR-10 dataset using standard data augmentation and a learning rate scheduler.

## 📌 Key Features
- Expert Routing Mechanism: Each Expert Block computes attention scores to dynamically combine outputs from multiple expert convolutional paths.
- Compact & Modular Design: Structured into a stem, backbone, and classifier, with two stacked Expert Blocks in the backbone.

### Training Enhancements:
- Data augmentation: random horizontal flip & crop.
- Regularization with dropout.
- Adaptive learning rate via StepLR.

## 🧱 Model Architecture
### Stem Layer: Conv → BatchNorm → ReLU (increases input channels from 3 to 64)

### Backbone:
- 2 × ExpertBlock(64 → 128)
- Each ExpertBlock contains:

### Global pooling & FC layers for routing.
- 4 expert conv branches combined via soft attention.

### Classifier: AdaptiveAvgPool → Flatten → Dropout(0.2) → Linear(128 → 10)

## 🧪 Dataset
- CIFAR-10: 60,000 32×32 color images in 10 classes.
### Transformations:
- Training: Random crop, horizontal flip, normalization.
- Testing: Normalization only.

## 🛠️ Training Details
- Hyperparameter	Value
- Optimizer	Adam
- Learning Rate	0.001
- Scheduler	StepLR (γ=0.5, step=20)
- Epochs	100
- Batch Size	128 (train), 100 (test)
- Dropout	0.2
- Experts per Block	4
- Reduction Ratio	4

## 📈 Results
- Final Test Accuracy: ~80–85%
- Training and validation curves show steady improvement, benefiting from the expert routing, dropout, and learning rate scheduling.

## 📊 Output Plots
- Loss Curve: Shows consistent convergence across epochs.
- Accuracy Curve: Gradual improvement with occasional plateaus.

## ▶️ Running the Code
```
# Ensure dependencies are installed
pip install torch torchvision matplotlib

# Run the training script (Python 3 recommended)
python custom_cnn_expert.py
```
The script will automatically download CIFAR-10, train the model, and display the accuracy/loss plots.

## 💡 Future Work
- Add more Expert Blocks for deeper networks.
- Try other routing mechanisms or attention strategies.
- Incorporate more aggressive data augmentation or regularization.

## 📚 Dependencies
- Python 3.7+
- PyTorch
- torchvision
- matplotlib
