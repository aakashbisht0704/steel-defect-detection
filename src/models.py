"""
Model architectures for Industrial Defect Detection.
Includes:
- ResNet-50 (Transfer Learning with 2-stage fine-tuning)
- EfficientNet-B0 (Modern compound scaled CNN)
- BaselineCNN (Lightweight 3-layer CNN trained from scratch)
- ConvAutoencoder (Unsupervised reconstruction anomaly detection)
"""
import torch
import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes: int = 6, freeze_backbone: bool = True):
    """
    ResNet-50 with pretrained weights and custom classification head.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
            
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model

def unfreeze_resnet_top_layers(model):
    """
    Unfreezes layer3 and layer4 for fine-tuning.
    """
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    return model

def get_efficientnet_b0(num_classes: int = 6, pretrained: bool = True):
    """
    EfficientNet-B0 transfer learning model.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

class BaselineCNN(nn.Module):
    """
    Standard Convolutional Neural Network trained from scratch.
    Serves as an essential benchmark baseline for model comparison.
    """
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 28x28
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(self.features(x))

class ConvAutoencoder(nn.Module):
    """
    Unsupervised Convolutional Autoencoder.
    Trained on defect-free surfaces; high reconstruction error signals an anomaly.
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 56x56
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 28x28
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),   # 224x224
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
