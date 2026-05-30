import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


# --- PROTOCOLO DE REPRODUCIBILIDAD ---
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PREPROCESAMIENTO (Página 5 del PDF) ---
# "apply the standard ImageNet normalisation to all images... resize to 224x224"
# "Apply random horizontal flip and colour jitter during training as data augmentation."
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# --- CARGA DE DATOS SEGlÚN EL PRESUPUESTO (Página 4 y 8) ---
# "50 labelled images per class for training (300 total for 6 classes)"
# "Split the source data into 50 train / 50 validation / remaining test images per class"
def get_few_shot_loaders(data_dir, batch_size=32):
    full_dataset = datasets.ImageFolder(root=data_dir)
    classes = full_dataset.classes
    targets = np.array(full_dataset.targets)

    train_idx, val_idx, test_idx = [], [], []
    for class_idx in range(len(classes)):
        class_indices = np.where(targets == class_idx)[0]
        np.random.shuffle(class_indices)
        train_idx.extend(class_indices[:50])
        val_idx.extend(class_indices[50:100])
        test_idx.extend(class_indices[100:])

    train_loader = DataLoader(Subset(datasets.ImageFolder(root=data_dir, transform=train_transform), train_idx),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(datasets.ImageFolder(root=data_dir, transform=val_test_transform), val_idx),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(datasets.ImageFolder(root=data_dir, transform=val_test_transform), test_idx),
                             batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# =====================================================================
# LISTING 1: COPIADO TAL CUAL DEL PDF (Con correcciones de sintaxis)
# =====================================================================

# 1. Feature extraction (frozen backbone)
model = models.resnet50(weights='IMAGENET1K_V2')  # Corregido error de OCR 'IMAGENETIK_V2'

for param in model.parameters():
    param.requires_grad = False  # freeze all layers

num_classes = 6
model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)


# 2. Fine-tuning: unfreeze last two residual blocks
# NOTA: En tu pipeline real, ejecutarás primero la extracción de características,
# y luego usarás este bloque para continuar con el Fine-Tuning.
def configurar_fine_tuning(model):
    for name, param in model.named_parameters():
        if 'layer3' in name or 'layer4' in name or 'fc' in name:
            param.requires_grad = True

    optimizer_ft = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    return optimizer_ft


# 3. Training loop (same for both strategies)
criterion = nn.CrossEntropyLoss()
num_epochs = 25  # Definido en el espacio de búsqueda hyperparámetros (20-30)


def train_model(model, train_loader, optimizer, num_epochs, criterion):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{num_epochs}] completada.")


# =====================================================================

if __name__ == '__main__':
    print(f"Modelo ResNet-50 cargado en {device} siguiendo las especificaciones del PDF.")
    # Descomentar una vez tengas las imágenes en las carpetas:
    # train_loader, val_loader, test_loader = get_few_shot_loaders("./data/DomainNet/real")
    # print("Entrenando Feature Extraction...")
    # train_model(model, train_loader, optimizer, num_epochs, criterion)