import torch
import torch.nn as nn
import torchvision.models as models


def get_model(num_classes=6, mode='feature_extraction'):
    """
    Configura el modelo ResNet-50 según la estrategia de Transfer Learning.
    Modos: 'feature_extraction' o 'fine_tuning'
    """
    # Cargamos pesos preentrenados actualizados de ImageNet
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)

    if mode == 'feature_extraction':
        # 1. Extracción de características: Congelamos todo el backbone
        for param in model.parameters():
            param.requires_grad = False

        # Reemplazamos la cabeza de clasificación (fully connected layer)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif mode == 'fine_tuning':
        # 2. Fine-tuning: Descongelamos los bloques residuales superiores (layer3 y layer4) y fc
        for name, param in model.named_parameters():
            if 'layer3' in name or 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Si vas a entrenar fine-tuning desde cero, recuerda cambiar la fc
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model


# Ejemplo de optimizadores según la documentación:
model_fe = get_model(num_classes=6, mode='feature_extraction')
optimizer_fe = torch.optim.Adam(model_fe.fc.parameters(), lr=1e-3)

model_ft = get_model(num_classes=6, mode='fine_tuning')
# Solo pasamos al optimizador los parámetros que requieren gradiente
optimizer_ft = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_ft.parameters()),
    lr=1e-4
)