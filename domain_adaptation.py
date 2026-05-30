import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

# --- CONFIGURACIÓN DE DISPOSITIVO Y REPRODUCIBILIDAD ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)  # Recordar reportar la media/std sobre 3 semillas distintas en la entrega


# =====================================================================
# LISTING 3: GRADIENT REVERSAL Y DANN (Copiado y completado del PDF)
# =====================================================================

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Recuperamos alpha guardado en el contexto
        alpha = ctx.saved_tensors[0].item()  # Corregido asignación rota del PDF
        return -alpha * grad_output, None  # Invierte el gradiente para el backbone


class DANNClassifier(nn.Module):
    def __init__(self, num_classes=6):  # Simplificado constructor para inicializar directo
        super().__init__()

        # Cargamos el backbone ResNet-50 sugerido
        resnet = models.resnet50(weights='IMAGENET1K_V2')
        num_features = resnet.fc.in_features

        # backbone.fc = nn.Identity() -> remove original head
        resnet.fc = nn.Identity()
        self.backbone = resnet

        # self.class_head = nn.Linear(backbone.fc.in_features, num_classes)
        self.class_head = nn.Linear(num_features, num_classes)

        # self.domain_head = nn.Sequential(...)
        self.domain_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Agregado para estabilidad numérica
            nn.Linear(256, 2)  # output de 2 clases: 0 para source (Real), 1 para target (Painting)
        )

    def forward(self, x, alpha=1.0):
        # feat = self.backbone(x)
        feat = self.backbone(x)

        # cls_out = self.class_head(feat)
        cls_out = self.class_head(feat)

        # rev = GradientReversal.apply(feat, alpha)
        rev = GradientReversal.apply(feat, alpha)

        # dom_out = self.domain_head(rev)
        dom_out = self.domain_head(rev)

        return cls_out, dom_out


# =====================================================================
# PIPELINES DE ENTRENAMIENTO PARA LAS 3 ESTRATEGIAS (Páginas 7 y 8)
# =====================================================================

# --- Estrategia 2: Target Fine-Tuning ---
# "collect a small labelled target-domain set (50 images per class) and fine-tune"
def train_target_finetuning(model, target_train_loader, criterion, lr=1e-5, epochs=15):
    print("Iniciando Estrategia 2: Target Fine-Tuning con presupuesto limitado...")
    model.to(device)

    # Solo descongelamos el último bloque residual (layer4) y la cabeza fc para cuidar los pesos
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        model.train()
        for images, labels in target_train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print("Target Fine-Tuning completado.")


# --- Estrategia 3: Style-Transfer Augmentation ---
# "augment the original 50-per-class source training set with the 30-per-class style-transferred images"
def train_style_augmentation(model, mixed_loader, criterion, lr=1e-4, epochs=30):
    print("Iniciando Estrategia 3: Style-Transfer Augmentation (Sin etiquetas extras del target)...")
    model.to(device)

    # Descongelamos bloques altos tal cual la parte A
    for name, param in model.named_parameters():
        if 'layer3' in name or 'layer4' in name or 'fc' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        model.train()
        for images, labels in mixed_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    print("Entrenamiento con Aumento de Datos Sintéticos completado.")


# --- Bucle de Pérdida Combinada para DANN (Opcional - Distinción) ---
def train_dann(dann_model, source_loader, target_loader, criterion, epochs=20, lambda_d=0.5):
    print("Iniciando Entrenamiento Adversarial de Dominio (DANN)...")
    dann_model.to(device)
    optimizer = torch.optim.Adam(dann_model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        dann_model.train()
        # Iteramos en paralelo sobre ambos dominios
        len_dataloader = min(len(source_loader), len(target_loader))
        target_iter = iter(target_loader)

        for i, (src_images, src_labels) in enumerate(source_loader):
            if i >= len_dataloader:
                break

            try:
                tgt_images, _ = next(target_iter)
            except StopIteration:
                break

            src_images, src_labels = src_images.to(device), src_labels.to(device)
            tgt_images = tgt_images.to(device)

            optimizer.zero_grad()

            # 1. Datos del dominio de origen (Source)
            src_cls_out, src_dom_out = dann_model(src_images, alpha=1.0)
            src_dom_labels = torch.zeros(src_images.size(0), dtype=torch.long).to(device)  # Clase 0: Real

            cls_loss = criterion(src_cls_out, src_labels)
            dom_loss_src = criterion(src_dom_out, src_dom_labels)

            # 2. Datos del dominio objetivo (Target - Pinturas sin etiquetas de clase)
            _, tgt_dom_out = dann_model(tgt_images, alpha=1.0)
            tgt_dom_labels = torch.ones(tgt_images.size(0), dtype=torch.long).to(device)  # Clase 1: Painting
            dom_loss_tgt = criterion(tgt_dom_out, tgt_dom_labels)

            # COMBINED LOSS (Ecuaciones de la página 8 del PDF)
            dom_loss = dom_loss_src + dom_loss_tgt
            loss = cls_loss + lambda_d * dom_loss

            loss.backward()
            optimizer.step()

    print("Entrenamiento DANN completado.")


if __name__ == '__main__':
    print("Módulo de Adaptación de Dominio (Parte C) cargado correctamente.")
    # Ejemplo de instanciación estructural:
    # model_dann = DANNClassifier(num_classes=6)