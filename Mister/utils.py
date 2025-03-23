import os
import csv
from datetime import datetime

def generar_nombre_archivo(config):
    """
    Genera un nombre de archivo CSV a partir de un diccionario de configuración.
    
    :param config: Diccionario con configuraciones para el nombre.
    :return: Nombre de archivo, e.g.: "clasificacion_jornadas_20250323.csv"
    """
    partes = []
    if "nombre" in config:
        partes.append(config["nombre"])
    if "fecha" in config:
        partes.append(config["fecha"])
    return "_".join(partes) + ".csv"

def aplanar_datos(diccionario):
    datos = []
    for registros in diccionario.values():
        datos.extend(registros)
    return datos

def obtener_temporada_actual():
    hoy = datetime.now()
    # Suponiendo que la temporada empieza en julio y termina en junio:
    if hoy.month >= 8:
        inicio = hoy.year % 100
        fin = (hoy.year + 1) % 100
    else:
        inicio = (hoy.year - 1) % 100
        fin = hoy.year % 100
    return f"{inicio:02d}/{fin:02d}"

def añadir_temporada(datos_por_jornada):
    temporada = obtener_temporada_actual()
    # datos_por_jornada es un diccionario, donde cada valor es una lista de registros
    for jornada, registros in datos_por_jornada.items():
        for registro in registros:
            registro["Temporada"] = temporada
    return datos_por_jornada

def guardar_en_csv(datos_list, base_path, filename_config, fieldnames=None):
    """
    Guarda en CSV una lista de diccionarios en una ruta dada con un nombre generado dinámicamente.
    
    :param datos_list: Lista de diccionarios a guardar.
    :param base_path: Ruta base donde se guardará el CSV (ejemplo: "./data/csv")
    :param filename_config: Diccionario de configuración para el nombre del archivo.
    :param fieldnames: Opcional, lista de columnas. Si es None, se infiere la unión de todas las claves.
    """
    os.makedirs(base_path, exist_ok=True)
    nombre_archivo = generar_nombre_archivo(filename_config)
    ruta_completa = os.path.join(base_path, nombre_archivo)

    if fieldnames is None:
        all_keys = set()
        for dic in datos_list:
            all_keys.update(dic.keys())
        fieldnames = list(all_keys)

    with open(ruta_completa, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in datos_list:
            writer.writerow(row)

    print(f"Archivo CSV guardado correctamente en: {ruta_completa}")
