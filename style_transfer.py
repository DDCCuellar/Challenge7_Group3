import torch
import torch.nn as nn
import torchvision.models as models


def gram_matrix(feat):
    """Calcula la matriz de Gram normalizada para capturar texturas."""
    b, c, h, w = feat.size()
    feat = feat.view(b, c, h * w)
    # Multiplicación matricial por batches
    gram = torch.bmm(feat, feat.transpose(1, 2))
    return gram / (c * h * w)


class StyleContentLoss(nn.Module):
    def __init__(self, vgg, content_layers, style_layers):
        super().__init__()
        self.vgg = vgg
        self.content_layers = content_layers
        self.style_layers = style_layers

        # Congelar el extractor VGG-19
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x, content_targets, style_targets):
        content_loss = 0.0
        style_loss = 0.0

        # Pasada hacia adelante capa por capa por la VGG
        for name, layer in self.vgg.features.named_children():
            x = layer(x)

            if name in self.content_layers:
                content_loss += nn.functional.mse_loss(x, content_targets[name])

            if name in self.style_layers:
                style_loss += nn.functional.mse_loss(gram_matrix(x), gram_matrix(style_targets[name]))

        return content_loss, style_loss


def run_style_transfer(content_img, style_img, num_steps=300, alpha=1.0, beta=1e4):
    """
    Ejecuta la optimización pixel a pixel para generar la imagen sintetizada.
    """
    device = content_img.device
    # Cargar VGG19 para extraer características de estilo/contenido
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

    content_layers = {'21': 'relu4_2'}  # Capa profunda para estructura
    style_layers = {'1': 'relu1_1', '6': 'relu2_1', '11': 'relu3_1', '20': 'relu4_1', '29': 'relu5_1'}

    # Extraer características estáticas del objetivo
    def get_features(img, layers_dict):
        feats = {}
        features_xt = img.clone()
        for name, layer in vgg.named_children():
            features_xt = layer(features_xt)
            if name in layers_dict:
                feats[layers_dict[name]] = features_xt.clone()
        return feats

    content_targets = get_features(content_img, {k: v for k, v in content_layers.items()})
    style_targets = get_features(style_img, {k: v for k, v in style_layers.items()})

    # La imagen generada empieza siendo una copia de la de contenido y requiere gradiente
    generated = content_img.clone().requires_grad_(True)

    # Invertimos los nombres para usarlos en el forward del loss
    loss_module = StyleContentLoss(vgg, list(content_layers.values()), list(style_layers.values()))
    optimizer = torch.optim.LBFGS([generated], lr=1.0, max_iter=20)

    step = 0
    while step < num_steps:
        def closure():
            nonlocal step
            optimizer.zero_grad()
            c_loss, s_loss = loss_module(generated, content_targets, style_targets)

            total_loss = alpha * c_loss + beta * s_loss
            total_loss.backward()

            if step % 50 == 0:
                print(f"Step {step}: Content Loss: {c_loss.item():.4f}, Style Loss: {s_loss.item():.4f}")
            step += 1
            return total_loss

        optimizer.step(closure)

    # Asegurar que los píxeles queden en el rango válido [0, 1] antes de retornar
    return torch.clamp(generated, 0.0, 1.0)