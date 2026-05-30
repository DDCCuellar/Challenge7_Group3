import os


def configurar_directorios():
    # Definimos la ruta base del proyecto
    base_dir = "."

    # Definimos las subcarpetas requeridas para el Grupo 3
    carpetas = [
        os.path.join(base_dir, "data", "DomainNet", "real"),
        os.path.join(base_dir, "data", "DomainNet", "painting"),
        os.path.join(base_dir, "data", "synthetic_target"),
        os.path.join(base_dir, "checkpoints"),
        os.path.join(base_dir, "figures")
    ]

    # Creamos cada carpeta si no existe
    for carpeta in carpetas:
        os.makedirs(carpeta, exist_ok=True)
        print(f"Directorio listo: {carpeta}")


# Ejecutar la función
configurar_directorios()