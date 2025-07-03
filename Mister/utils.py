import os
import csv
from datetime import datetime
from configparser import ConfigParser
from contextlib import contextmanager
import psycopg2
import hashlib
import unicodedata
import re
import pandas as pd

def añadir_hash(df, schema='dbo', tabla=''):
    if not tabla:
        raise ValueError("Para añadir_hash con orden de tabla real, debes pasar el nombre de la tabla.")

    columnas_excluir = {'f_carga', 'hash'}  # puedes excluir más si lo deseas

    with conexion_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
            """, (schema, tabla))
            columnas_ordenadas = [row[0] for row in cur.fetchall() if row[0] not in columnas_excluir]

    # Eliminar columnas que no existan en el DataFrame (por si acaso)
    columnas_validas = [col for col in columnas_ordenadas if col in df.columns]

    def calcular_hash(row):
        cadena = ''.join(str(row[col]) for col in columnas_validas)
        return hashlib.sha256(cadena.encode('utf-8')).hexdigest()

    df['hash'] = df.apply(calcular_hash, axis=1)
    return df

def añadir_f_carga(df):
    log("añadir_f_carga: Iniciamos a añadir f_carga")
    df["f_carga"] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    log("añadir_f_carga: f_carga añadida")
    return df

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

def aplanar_datos(data):
    """
    Aplana estructuras comunes de datos en una lista de diccionarios.
    - Si recibe una lista de dicts, la devuelve tal cual.
    - Si recibe un dict con listas de dicts como valores, las concatena.
    - Si recibe un solo dict, lo mete en una lista.
    """
    if isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            return data
        else:
            raise TypeError("La lista debe contener solo diccionarios")

    elif isinstance(data, dict):
        if all(isinstance(v, list) for v in data.values()):
            # Caso tipo: {1: [dict, dict], 2: [dict, dict]}
            resultado = []
            for lista in data.values():
                if all(isinstance(item, dict) for item in lista):
                    resultado.extend(lista)
                else:
                    raise TypeError("Las listas dentro del dict deben contener solo diccionarios")
            return resultado

        elif all(isinstance(v, (str, int, float)) for v in data.values()):
            # Caso tipo: {"a": 1, "b": 2}
            return [data]

    raise TypeError("Se esperaba una lista de diccionarios o un dict con listas de diccionarios")


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
    nombre_archivo = filename_config["archivo"]
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
    file_name = f"./log/log_{now.strftime('%Y%m%d')}.txt"
    
    # Abrir el fichero en modo "append" para no sobreescribir los logs del mismo día
    with open(file_name, "a", encoding="utf-8") as f:
        f.write(log_line)

def leer_config_db(archivo='config.ini', seccion='postgresql'):
    """Lee la configuración de la base de datos desde config.ini"""
    ruta_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), archivo)
    log(f"leer_config_db: Buscando config.ini en {os.path.abspath(archivo)}")
    parser = ConfigParser()
    archivos_leidos = parser.read(ruta_config)
    log(f"conexion_db: Archivos leídos: {archivos_leidos}")


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

def get_base_path_from_ini(archivo='config.ini', seccion='paths', clave='base_csv'):
    config = ConfigParser()
    config.read("config.ini")
    return config.get("paths", "base_csv", fallback="./data/csv")

def obtener_urls_desde_db(schema="dbo", tabla="urls_jugadores"):
    """
    Devuelve una lista de URLs de jugadores desde la tabla de PostgreSQL.
    """
    log(f"obtener_urls_desde_db: Cargando URLs desde {schema}.{tabla}")
    urls = []

    try:
        with conexion_db() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT url FROM {schema}.{tabla}")
                urls = [row[0] for row in cur.fetchall()]
                log(f"obtener_urls_desde_db: Se obtuvieron {len(urls)} URLs")
    except Exception as e:
        log(f"obtener_urls_desde_db: Error al obtener URLs desde PostgreSQL - {e}")

    return urls

def limpiar_columna(col):
    # Normaliza (quita tildes), reemplaza espacios por _, quita caracteres raros
    col = unicodedata.normalize("NFD", col)
    col = col.encode("ascii", "ignore").decode("utf-8")  # elimina tildes
    col = col.lower().replace(" ", "_")
    return col