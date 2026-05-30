import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os

# --- CONFIGURACIÓN DE DISPOSITIVO Y SEMILLAS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# --- PREPROCESAMIENTO IDÉNTICO (Páginas 5 y 10) ---
# "Always apply the same ImageNet normalisation parameters to both domains"
imsize = 256  # Tamaño sugerido por la guía para un cálculo práctico
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

unloader = transforms.Compose([
    # Operación inversa para poder guardar la imagen como un archivo PNG válido
    transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.ToPILImage()
])


def load_image(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)  # Añadir dimensión de batch (b=1)
    return image.to(device, torch.float)


# =====================================================================
# LISTING 2: NEURAL STYLE TRANSFER (Copiado y completado desde el PDF)
# =====================================================================

def gram_matrix(feat):
    b, c, h, w = feat.size()
    feat = feat.view(b, c, h * w)  # Corregido error tipográfico del PDF (hw -> h * w)
    return torch.bmm(feat, feat.transpose(1, 2)) / (c * h * w)


class StyleContentLoss(nn.Module):
    def __init__(self, vgg, content_layers, style_layers):
        super().__init__()
        self.vgg = vgg
        self.content_layers = content_layers
        self.style_layers = style_layers

    def forward(self, x, content_targets, style_targets):  # Nota: 'style_targets' en plural para consistencia
        content_loss, style_loss = 0.0, 0.0
        for name, layer in self.vgg.features.named_children():
            x = layer(x)
            if name in self.content_layers:
                content_loss += nn.functional.mse_loss(x, content_targets[name])
            if name in self.style_layers:
                style_loss += nn.functional.mse_loss(gram_matrix(x), style_targets[name])
        return content_loss, style_loss


# --- FUNCIÓN PRINCIPAL DE OPTIMIZACIÓN (L-BFGS) ---
def run_style_transfer(content_image, style_image, num_steps=300, alpha=1.0, beta=1e4):
    # Cargar extractor de características VGG-19 preentrenado (Páginas 5 y 6 del PDF)
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).to(device).eval()

    # Congelar los parámetros de VGG para no optimizar la red
    for param in vgg.parameters():
        param.requires_grad = False

    # Capas sugeridas en el documento (Página 6):
    # Content: relu4_2 (mapeada internamente en VGG19 de torchvision como la capa '21')
    # Style: relu1_1 ('1'), relu2_1 ('6'), relu3_1 ('11'), relu4_1 ('20'), relu5_1 ('29')
    content_layers_map = {'21': 'relu4_2'}
    style_layers_map = {'1': 'relu1_1', '6': 'relu2_1', '11': 'relu3_1', '20': 'relu4_1', '29': 'relu5_1'}

    # Extraer las activaciones estáticas del objetivo (Target features)
    def extract_features(img):
        c_feats, s_feats = {}, {}
        features_xt = img.clone()
        for name, layer in vgg.features.named_children():
            features_xt = layer(features_xt)
            if name in content_layers_map:
                c_feats[content_layers_map[name]] = features_xt.clone()
            if name in style_layers_map:
                s_feats[style_layers_map[name]] = features_xt.clone()
        return c_feats, s_feats

    content_targets, _ = extract_features(content_image)
    _, style_targets = extract_features(style_image)

    # Inicializar la imagen generada clonando la de contenido (Página 6 del PDF)
    generated = content_image.clone().requires_grad_(True)

    # Configurar el optimizador L-BFGS tal cual pide la guía
    optimizer = torch.optim.LBFGS([generated], lr=1.0, max_iter=20)

    # Instanciar el módulo de pérdidas usando los nombres de las capas internas
    loss_model = StyleContentLoss(vgg, list(content_layers_map.values()), list(style_layers_map.values()))

    print("Iniciando optimización de píxeles...")

    # Implementación del bucle con la función closure() requerida por L-BFGS
    step = 0
    while step < num_steps:
        def closure():
            nonlocal step
            optimizer.zero_grad()
            c_loss, s_loss = loss_model(generated, content_targets, style_targets)

            # Ecuación (1): L_total = alpha * L_content + beta * L_style
            loss = alpha * c_loss + beta * s_loss
            loss.backward()

            if step % 50 == 0:
                print(
                    f"Step {step:03d} -> Total Loss: {loss.item():.4f} | Content Loss: {c_loss.item():.4f} | Style Loss: {s_loss.item():.4f}")

            step += 1
            return loss

        optimizer.step(closure)

    # Retornar la imagen resultante asegurando que no se salga del rango estándar tras des-normalizar
    return generated.detach()



# =====================================================================

def save_output(tensor, path):
    """Guarda el tensor procesado como una imagen de disco PNG."""
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image)
    image.save(path)


if __name__ == '__main__':
    print("Módulo de transferencia de estilo neurónal (Parte B) cargado correctamente.")
    # Ejemplo de uso estructural cuando tengas listos los archivos:
    # content_img = load_image("./data/DomainNet/real/beach/imagen_ejemplo.jpg")
    # style_img = load_image("./data/DomainNet/painting/beach/pintura_ejemplo.jpg")
    # output = run_style_transfer(content_img, style_img)
    # save_output(output, "./data/synthetic_target/beach/synthetic_01.png")