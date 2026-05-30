import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.models as models


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # En la propagación hacia atrás, invertimos el gradiente multiplicándolo por -alpha
        alpha = ctx.saved_tensors[0].item()
        return -alpha * grad_output, None


class DANNClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Usamos ResNet-50 como extractor de características base
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_features = resnet.fc.in_features

        # Removemos la capa fc original convirtiéndola en Identidad
        resnet.fc = nn.Identity()
        self.backbone = resnet

        # Cabeza clasificadora de tareas (Clases de paisajes)
        self.class_head = nn.Linear(num_features, num_classes)

        # Cabeza clasificadora de dominio (0: Real vs 1: Painting)
        self.domain_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x, alpha=1.0):
        # 1. Extraer características vectoriales
        feat = self.backbone(x)

        # 2. Clasificación de clases del objeto
        cls_out = self.class_head(feat)

        # 3. Clasificación de dominio con reversión de gradiente
        rev_feat = GradientReversal.apply(feat, alpha)
        dom_out = self.domain_head(rev_feat)

        return cls_out, dom_out