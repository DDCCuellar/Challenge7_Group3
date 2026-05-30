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
    # 1. Aplanamos las dimensiones de espacio (H y W) en una sola dimensión continua (H * W)
    feat = feat.view(b, c, h * w)

    # 2. Multiplicamos la matriz por su propia transpuesta en sus dimensiones indexadas
    # feat: [b, c, h*w] -> feat.transpose(1, 2): [b, h*w, c]
    # El resultado de torch.bmm será un tensor de tamaño [b, c, c] (64x64 para la primera capa)
    gram = torch.bmm(feat, feat.transpose(1, 2))

    # 3. Dividimos de forma estricta por el producto escalar (C * H * W) como dicta la Ec. 2 del PDF
    return gram / (c * h * w)

class StyleContentLoss(nn.Module):
    def __init__(self, vgg, content_layers, style_layers):
        super().__init__()
        self.vgg = vgg
        self.content_layers = content_layers
        self.style_layers = style_layers

    def forward(self, x, content_targets, style_targets):
        content_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        style_loss = torch.tensor(0.0, device=x.device, requires_grad=True)

        # Copiamos la imagen inicial para pasarla capa por capa por la VGG-19
        features_xt = x.clone().contiguous()

        for name, layer in self.vgg.features.named_children():
            features_xt = layer(features_xt)

            # ¡CORREGIDO!: Se cambiaron los "+" por "+=" para guardar el tensor acumulado
            if name in self.content_layers:
                content_loss = content_loss + nn.functional.mse_loss(features_xt, content_targets[name])
            if name in self.style_layers:
                style_loss = style_loss + nn.functional.mse_loss(gram_matrix(features_xt), style_targets[name])

        return content_loss, style_loss

# --- FUNCIÓN PRINCIPAL DE OPTIMIZACIÓN (L-BFGS) ---
def run_style_transfer(content_image, style_image, num_steps=100, alpha=1.0, beta=1e4):
    # Cargar VGG-19 en modo evaluación y congelar parámetros
    # DESPUÉS
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    # Deshabilitar todos los ReLU in-place para evitar corrupción del grafo de gradientes
    for module in vgg.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    # Mapeo de capas sugeridas por el PDF
    content_layers_map = {'21': '21'}  # relu4_2 corresponde al índice '21' en torchvision VGG-19
    style_layers_map = {'1': '1', '6': '6', '11': '11', '20': '20', '29': '29'}  # relu1_1, 2_1, 3_1, 4_1, 5_1

    # Extraer las características estáticas del contenido y estilo antes de optimizar
    def extract_features(img, layers_dict):
        feats = {}
        features_xt = img.clone()
        for name, layer in vgg.features.named_children():
            features_xt = layer(features_xt)
            if name in layers_dict:
                feats[name] = features_xt.clone()
        return feats

    # DESPUÉS (correcto)
    content_targets = extract_features(content_image, content_layers_map)
    style_targets_raw = extract_features(style_image, style_layers_map)
    style_targets = {name: gram_matrix(feat) for name, feat in style_targets_raw.items()}

    # Inicializar la imagen generada a partir de la de contenido con gradientes activos
    generated = content_image.clone().requires_grad_(True)

    # El optimizador L-BFGS requiere aprender los píxeles de la imagen directamente
    optimizer = torch.optim.LBFGS([generated], lr=1.0, max_iter=20)

    # Instanciar el módulo de pérdidas corregido
    loss_model = StyleContentLoss(vgg, list(content_layers_map.keys()), list(style_layers_map.keys()))

    print("Iniciando optimización de píxeles...")

    # Contador de pasos mutable compatible con estructuras internas de closures
    step = [0]

    while step[0] < num_steps:
        def closure():
            optimizer.zero_grad()
            c_loss, s_loss = loss_model(generated, content_targets, style_targets)

            # Ecuación (1) del PDF: L_total = alpha * L_content + beta * L_style
            total_loss = (c_loss * alpha) + (s_loss * beta)

            total_loss.backward()

            if step[0] % 50 == 0:
                print(
                    f"Step {step[0]:03d} -> Total Loss: {total_loss.item():.4f} | Content Loss: {c_loss.item():.4f} | Style Loss: {s_loss.item():.4f}")

            step[0] += 1
            return total_loss

        optimizer.step(closure)

    return generated.detach()

# =====================================================================

def save_output(tensor, path):
    """Guarda el tensor procesado como una imagen de disco PNG."""
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image)
    image.save(path)


if __name__ == '__main__':
    print(f"Iniciando el pipeline automatizado para el Grupo 3 en {device}...")

    # 1. Definir las rutas base de tu estructura
    SRC_DIR = "./data/DomainNet/real"
    TGT_STYLE_DIR = "./data/DomainNet/painting"
    OUTPUT_BASE_DIR = "./data/synthetic_target"

    # 2. Las 6 clases asignadas exclusivamente al Grupo 3
    clases_grupo3 = ["beach", "bridge", "forest", "mountain", "river", "tree"]

    # Presupuesto solicitado por el documento: 30 imágenes por clase = 180 imágenes en total
    IMAGENES_POR_CLASE = 30

    for clase in clases_grupo3:
        print(f"\n=========================================")
        print(f" PROCESANDO CATEGORÍA: {clase.upper()}")
        print(f"=========================================")

        # Rutas específicas de la clase para origen, estilo y destino
        clase_src_dir = os.path.join(SRC_DIR, clase)
        clase_style_dir = os.path.join(TGT_STYLE_DIR, clase)
        clase_out_dir = os.path.join(OUTPUT_BASE_DIR, clase)

        # Crear automáticamente la carpeta de salida para la clase si no existe
        os.makedirs(clase_out_dir, exist_ok=True)

        # Validar que las carpetas de origen existan y tengan imágenes
        if not os.path.exists(clase_src_dir) or not os.path.exists(clase_style_dir):
            print(f"Error: No se encontraron las carpetas de DomainNet para la clase '{clase}'.")
            print(f"Asegúrate de haber descargado y colocado los archivos en {clase_src_dir} y {clase_style_dir}")
            continue

        # Listar archivos de imágenes válidos (extensiones comunes)
        formatos_validos = ('.jpg', '.jpeg', '.png', '.bmp')
        fotos_reales = sorted([f for f in os.listdir(clase_src_dir) if f.lower().endswith(formatos_validos)])
        pinturas_estilo = sorted([f for f in os.listdir(clase_style_dir) if f.lower().endswith(formatos_validos)])

        # Verificar que tengamos suficientes imágenes para cumplir el presupuesto
        total_disponibles = min(len(fotos_reales), len(pinturas_estilo))
        if total_disponibles < IMAGENES_POR_CLASE:
            print(
                f"Advertencia: Solo hay {total_disponibles} pares de imágenes en '{clase}'. Se procesarán solo esas.")
            limite_proceso = total_disponibles
        else:
            limite_proceso = IMAGENES_POR_CLASE

        # 3. Bucle interno para generar las 30 imágenes sintéticas de la clase actual
        for i in range(limite_proceso):
            nombre_foto = fotos_reales[i]
            nombre_pintura = pinturas_estilo[i]  # Emparejamiento 1 a 1 directo

            ruta_contenido = os.path.join(clase_src_dir, nombre_foto)
            ruta_estilo = os.path.join(clase_style_dir, nombre_pintura)

            # Definir el nombre del archivo de salida según un estándar limpio
            nombre_salida = f"synthetic_{clase}_{i + 1:02d}.png"
            ruta_salida = os.path.join(clase_out_dir, nombre_salida)

            # Omitir si la imagen ya fue generada previamente (útil por si se corta la luz o el proceso)
            if os.path.exists(ruta_salida):
                print(f"[{i + 1}/{limite_proceso}] {nombre_salida} ya existe. Omitiendo...")
                continue

            print(f"\nGenerando [{i + 1}/{limite_proceso}]: {nombre_salida}")
            print(f"  -> Contenido (Real): {nombre_foto}")
            print(f"  -> Estilo (Painting): {nombre_pintura}")

            try:
                # Cargar los tensores normalizados en la GPU/CPU
                img_contenido = load_image(ruta_contenido)
                img_estilo = load_image(ruta_estilo)

                # Ejecutar la optimización de Gatys usando L-BFGS (300 pasos por imagen)
                # alpha=1.0, beta=1e4 es la relación sugerida en el PDF (ratio 1:10000)
                resultado_tensor = run_style_transfer(
                    img_contenido,
                    img_estilo,
                    num_steps=100,
                    alpha=1.0,
                    beta=1e4
                )

                # Desnormalizar y exportar a disco duro
                save_output(resultado_tensor, ruta_salida)
                print(f"Guardado con éxito en: {ruta_salida}")

            except Exception as e:
                print(f"Error al procesar el par {i + 1}: {str(e)}")

    print("\n=========================================")
    print(" Pipeline finalizado. ¡Tus 180 imágenes están listas!")
    print("=========================================")