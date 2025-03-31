import os
import csv
from datetime import datetime
from configparser import ConfigParser
from contextlib import contextmanager
import psycopg2
import hashlib
import json

def añadir_hash(lista_dicts):
    """
    Añade un campo 'hash' a cada diccionario de la lista, generado con json.dumps ordenado.
    """
    log("añadir_hash: Iniciamos a añadir hash")
    if not isinstance(lista_dicts, list):
        raise TypeError("Se esperaba una lista de diccionarios")

    for item in lista_dicts:
        if not isinstance(item, dict):
            raise TypeError("Cada elemento debe ser un diccionario")

        # Serialización ordenada y estable
        item_serializado = json.dumps(item, sort_keys=True, ensure_ascii=False)
        item["hash"] = hashlib.sha256(item_serializado.encode("utf-8")).hexdigest()

    log("añadir_hash: hash añadido")
    return lista_dicts

def añadir_f_carga(lista_registros):
    log(f"añadir_f_carga: Iniciamos a añadir f_carga")
    f_carga_actual = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if isinstance(lista_registros, list):
        for registro in lista_registros:
            if isinstance(registro, dict):
                registro["f_carga"] = f_carga_actual
    log(f"añadir_f_carga: f_carga añadida")
    return lista_registros

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

def aplanar_datos(datos):
    """
    Si 'datos' es un diccionario, lo aplana a una lista.
    Si ya es una lista, lo devuelve tal cual.
    """
    if isinstance(datos, dict):
        resultado = []
        for registros in datos.values():
            resultado.extend(registros)
        return resultado
    elif isinstance(datos, list):
        return datos
    else:
        raise TypeError("Se esperaba un dict o una lista")


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

def añadir_temporada(datos):
    log(f"añadir_temporada: Iniciamos a añadir la temporada")
    temporada = obtener_temporada_actual()
    if isinstance(datos, dict):
        # Si es un diccionario, se asume que cada valor es una lista de registros
        for key, registros in datos.items():
            for registro in registros:
                registro["Temporada"] = temporada
    elif isinstance(datos, list):
        # Si es una lista, cada elemento es un registro
        for registro in datos:
            registro["Temporada"] = temporada
    else:
        raise TypeError("El formato de datos no es soportado (se esperaba dict o list)")
    log(f"añadir_temporada: temporada añadida")
    return datos

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

    log(f"Archivo CSV guardado correctamente en: {ruta_completa}")

def log(message):
    # Obtener el momento actual
    now = datetime.now()
    # Formatear el timestamp: año-mes-dia hora:minutos:segundos.milisegundos
    log_time = now.strftime("%Y-%m-%d %H:%M:%S") + f".{now.microsecond//1000:03d}"
    # Crear la línea de log
    log_line = f"{log_time} - {message}\n"
    
    # Asegurarse de que el directorio de log existe
    os.makedirs("log", exist_ok=True)
    
    # Construir el nombre del fichero usando el formato añomesdia (por ejemplo: log_20250328.txt)
    file_name = f"log/log_{now.strftime('%Y%m%d')}.txt"
    
    # Abrir el fichero en modo "append" para no sobreescribir los logs del mismo día
    with open(file_name, "a", encoding="utf-8") as f:
        f.write(log_line)

def leer_config_db(archivo='Mister/config.ini', seccion='postgresql'):
    """Lee la configuración de la base de datos desde config.ini"""
    log(f"leer_config_db: Buscando config.ini en {os.path.abspath(archivo)}")
    parser = ConfigParser()
    parser.read(archivo)
    log(f"conexion_db: Archivos leídos: {parser.read(archivo)}")

    if parser.has_section(seccion):
        return {param[0]: param[1] for param in parser.items(seccion)}
    else:
        raise Exception(f'Sección {seccion} no encontrada en el archivo {archivo}')

@contextmanager
def conexion_db():
    """Context manager para conexión segura y limpia a PostgreSQL"""
    conn = None
    try:
        config = leer_config_db()
        conn = psycopg2.connect(**config)
        log("conexion_db: Conexión establecida correctamente")
        yield conn
    except Exception as e:
        import traceback
        error_type = type(e).__name__
        log(f"conexion_db: Error durante la conexión - {error_type}: {e}")
        raise
    finally:
        if conn:
            conn.close()
            log("conexion_db: Conexión cerrada correctamente")

