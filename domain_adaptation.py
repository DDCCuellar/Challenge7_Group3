import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Function
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset

# --- CONFIGURACIÓN DE DISPOSITIVO Y PROTOCOLO DE REPRODUCIBILIDAD ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- PIPELINE DE PREPROCESAMIENTO IDÉNTICO (Páginas 5 y 9) ---
# "Use identical preprocessing... to ensure the domain shift is not artefactual."
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# =====================================================================
# LISTING 3: GRADIENT REVERSAL Y ARQUITECTURA DANN (Corregido del PDF)
# =====================================================================

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha, device=x.device))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.saved_tensors[0].item()
        # Invierte el gradiente multiplicándolo por -alpha para el backbone
        return -alpha * grad_output, None


class DANNClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Usamos el backbone ResNet-50 oficial
        resnet = models.resnet50(weights='IMAGENET1K_V2')
        num_features = resnet.fc.in_features

        resnet.fc = nn.Identity()  # Remove original head (Línea 24 del PDF)
        self.backbone = resnet

        self.class_head = nn.Linear(num_features, num_classes)
        self.domain_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # 2 clases: 0 = Source (Real), 1 = Target (Painting)
        )

    def forward(self, x, alpha=1.0):
        feat = self.backbone(x)
        cls_out = self.class_head(feat)
        rev = GradientReversal.apply(feat, alpha)
        dom_out = self.domain_head(rev)
        return cls_out, dom_out


# =====================================================================
# MÉTODOS DE EVALUACIÓN Y ENTRENAMIENTO PARA LAS ESTRATEGIAS
# =====================================================================

def evaluar_modelo(model, dataloader, nombre_split="Test"):
    """Calcula la precisión (Accuracy) del modelo en un conjunto de datos."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            # DANN devuelve una tupla (class, domain), ResNet común solo devuelve logits
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy en {nombre_split}: {accuracy:.2f}%")
    return accuracy


# --- Estrategia 2: Target Fine-Tuning ---
# "collect a small labelled target-domain set (50 images per class) and fine-tune"
def train_target_finetuning(model, target_train_loader, criterion, lr=1e-5, epochs=15):
    print("\n[Estrategia 2] Iniciando Target Fine-Tuning (Presupuesto limitado)...")

    # Congelamos los bloques bajos y solo dejamos activos layer4 y la cabeza fc
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name or 'class_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        model.train()
        loss_acumulada = 0.0
        for images, labels in target_train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_acumulada += loss.item()
        print(f"  Epoch [{epoch + 1}/{epochs}] -> Loss: {loss_acumulada / len(target_train_loader):.4f}")


# --- Estrategia 3: Style-Transfer Augmentation ---
# "augment the original 50-per-class source training set with the 30-per-class style-transferred images"
def train_style_augmentation(model, mixed_loader, criterion, lr=1e-4, epochs=25):
    print("\n[Estrategia 3] Iniciando Style-Transfer Augmentation (Datos Sintéticos)...")

    # Activamos los bloques residuales altos para asimilar las texturas mixtas
    for name, param in model.named_parameters():
        if 'layer3' in name or 'layer4' in name or 'fc' in name or 'class_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(epochs):
        model.train()
        loss_acumulada = 0.0
        for images, labels in mixed_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_acumulada += loss.item()
        print(f"  Epoch [{epoch + 1}/{epochs}] -> Loss: {loss_acumulada / len(mixed_loader):.4f}")


# =====================================================================
# EJECUCIÓN DEL PROTOCOLO DE EXPERIMENTACIÓN
# =====================================================================

if __name__ == '__main__':
    # Configurar semilla única para la ejecución del script activo
    set_seed(42)

    # Directorios de datos organizados en las partes anteriores
    REAL_DIR = "./data/DomainNet/real"
    PAINTING_DIR = "./data/DomainNet/painting"
    SYNTHETIC_DIR = "./data/synthetic_target"

    # 1. GENERACIÓN DE LOGISTICS DE DATOS (Asegurando el presupuesto Few-Shot)
    dataset_real = datasets.ImageFolder(root=REAL_DIR)
    targets_real = np.array(dataset_real.targets)
    clases = dataset_real.classes

    idx_src_train, idx_src_test = [], []
    idx_tgt_train, idx_tgt_test = [], []

    # Particionado controlado de 50 imágenes por clase según directrices del PDF
    for c in range(len(clases)):
        indices_c = np.where(targets_real == c)[0]
        np.random.shuffle(indices_c)
        idx_src_train.extend(indices_c[:50])  # Fuente de entrenamiento original (300 fotos)
        idx_src_test.extend(indices_c[50:])  # Conjunto de prueba del dominio origen

    # Cargador de Entrenamiento Origen (Fotos Reales)
    src_train_loader = DataLoader(Subset(datasets.ImageFolder(root=REAL_DIR, transform=train_transform), idx_src_train),
                                  batch_size=32, shuffle=True)
    src_test_loader = DataLoader(
        Subset(datasets.ImageFolder(root=REAL_DIR, transform=val_test_transform), idx_src_test), batch_size=32,
        shuffle=False)

    # Cargar el Dominio Objetivo de pinturas reales para entrenamiento limitado (Estrategia 2) y Prueba final
    dataset_painting = datasets.ImageFolder(root=PAINTING_DIR)
    targets_painting = np.array(dataset_painting.targets)

    for c in range(len(clases)):
        indices_pt = np.where(targets_painting == c)[0]
        np.random.shuffle(indices_pt)
        idx_tgt_train.extend(indices_pt[:50])  # Presupuesto pequeño de anotación en Target (Estrategia 2)
        idx_tgt_test.extend(indices_pt[50:])  # Test definitivo para evaluar el Domain Shift

    tgt_train_loader = DataLoader(
        Subset(datasets.ImageFolder(root=PAINTING_DIR, transform=train_transform), idx_tgt_train), batch_size=32,
        shuffle=True)
    tgt_test_loader = DataLoader(
        Subset(datasets.ImageFolder(root=PAINTING_DIR, transform=val_test_transform), idx_tgt_test), batch_size=32,
        shuffle=False)

    # Cargar tus 180 imágenes sintéticas generadas en la Parte B
    try:
        dataset_synthetic = datasets.ImageFolder(root=SYNTHETIC_DIR, transform=train_transform)
        # Combinar las 300 fotos reales de entrenamiento + 180 imágenes estilizadas (Estrategia 3)
        dataset_mixto = ConcatDataset(
            [Subset(datasets.ImageFolder(root=REAL_DIR, transform=train_transform), idx_src_train), dataset_synthetic])
        mixed_loader = DataLoader(dataset_mixto, batch_size=32, shuffle=True)
        print(f"-> Dataset aumentado creado con éxito: {len(dataset_mixto)} imágenes totales.")
    except Exception as e:
        print("Nota: No se pudo cargar la carpeta sintética completa aún. Asegúrate de correr la Parte B antes.")
        mixed_loader = None

    # -----------------------------------------------------------------
    # EXPERIMENTO 1: MEDIR EL DOMAIN SHIFT PENALTY (Baseline)
    # -----------------------------------------------------------------
    print("\n=== MEDICIÓN DEL DOMAIN SHIFT PENALTY ===")

    # Recreamos la arquitectura de la Parte A para cargar sus pesos guardados
    base_model = models.resnet50()
    base_model.fc = nn.Linear(base_model.fc.in_features, 6)

    ruta_checkpoint = "./checkpoints/resnet50_feature_extraction.pth"
    if os.path.exists(ruta_checkpoint):
        base_model.load_state_dict(torch.load(ruta_checkpoint, map_view=device))
        base_model.to(device)
        print("-> Pesos de la Parte A cargados con éxito.")

        # Evaluar en origen vs destino para calcular la Ecuación (4)
        acc_source = evaluar_modelo(base_model, src_test_loader, "Source (Real Photos)")
        acc_target = evaluar_modelo(base_model, tgt_test_loader, "Target (Painting Benchmark)")

        # Delta shift = Acc_source - Acc_target
        delta_shift = acc_source - acc_target
        print(f"Domain Shift Penalty (\u0394_shift): {delta_shift:.2f}% de pérdida de precisión.")
    else:
        print("No se encontró el checkpoint de la Parte A en ./checkpoints/. Corre classifier.py primero.")

    # -----------------------------------------------------------------
    # EXPERIMENTO 2: EJECUTAR ADAPTACIONES (Si los loaders están listos)
    # -----------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(ruta_checkpoint):
        # Clonamos el modelo base para aplicar la Estrategia 2 sin alterar los pesos originales
        model_ft = models.resnet50()
        model_ft.fc = nn.Linear(model_ft.fc.in_features, 6)
        model_ft.load_state_dict(torch.load(ruta_checkpoint))
        model_ft.to(device)

        train_target_finetuning(model_ft, tgt_train_loader, criterion)
        print("Resultado tras Target Fine-Tuning:")
        evaluar_modelo(model_ft, tgt_test_loader, "Target Test (Estrategia 2)")


        # Clonamos de nuevo para evaluar el impacto de tus imágenes sintéticas (Estrategia 3)
        if mixed_loader is not None:
            model_aug = models.resnet50()
            model_aug.fc = nn.Linear(model_aug.fc.in_features, 6)
            model_aug.load_state_dict(torch.load(ruta_checkpoint))
            model_aug.to(device)

            train_style_augmentation(model_aug, mixed_loader, criterion)
            print("Resultado tras Style-Transfer Augmentation:")
            evaluar_modelo(model_aug, tgt_test_loader, "Target Test (Estrategia 3)")