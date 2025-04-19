# cnn_model.py
import torch.nn as nn
from torchvision import models

class ResNet50FineTuner(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5, dense_layer_size=512, freeze_fc=False, freeze_last_conv=False):
        super(ResNet50FineTuner, self).__init__()

        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Freeze layers if required
        if freeze_fc:
            for param in self.base_model.fc.parameters():
                param.requires_grad = False

        if freeze_last_conv:
            for name, param in self.base_model.named_parameters():
                if "layer4" in name:
                    param.requires_grad = False

        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, dense_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_layer_size, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
