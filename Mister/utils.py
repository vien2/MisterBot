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
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import gzip
import glob
import shlex
import shutil
import subprocess
from datetime import datetime, timedelta
import time
import threading

def añadir_hash(df, schema='dbo', tabla=''):
    columnas_excluir = {'f_carga', 'hash'}
    columnas_validas = [col for col in df.columns if col not in columnas_excluir]

    def calcular_hash(row):
        valores = []
        for col in columnas_validas:
            v = row[col]
            if pd.isna(v) or v is None:
                valores.append("")
            else:
                valores.append(str(v))
        cadena = "|".join(valores)
        return hashlib.sha256(cadena.encode("utf-8")).hexdigest()

    df["hash"] = df.apply(calcular_hash, axis=1)
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
    log("añadir_temporada: Iniciamos a añadir la temporada")
    temporada = obtener_temporada_actual()

    def set_temporada(registro):
        # normaliza todas las claves a minúscula para comparar
        if not any(k.lower() == "temporada" for k in registro.keys()):
            registro["temporada"] = temporada

    if isinstance(datos, dict):
        for key, registros in datos.items():
            for registro in registros:
                set_temporada(registro)
    elif isinstance(datos, list):
        for registro in datos:
            set_temporada(registro)
    else:
        raise TypeError("El formato de datos no es soportado (se esperaba dict o list)")

    log("añadir_temporada: temporada añadida (si faltaba)")
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

def obtener_urls_desde_db(schema, tabla="urls_jugadores"):
    """
    Devuelve una lista de URLs de jugadores desde la tabla de PostgreSQL.
    """
    temporada = obtener_temporada_actual()
    log(f"obtener_urls_desde_db: Cargando URLs desde {schema}.{tabla}")
    urls = []

    try:
        with conexion_db() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT url FROM {schema}.{tabla} WHERE temporada = '{temporada}'")
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

def cerrar_popup_anuncios(driver):
    """Cierra el popup de anuncios si aparece"""
    try:
        boton_cerrar = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "div#popup button.popup-close"))
        )
        driver.execute_script("arguments[0].click();", boton_cerrar)
        log("Popup de anuncios detectado y cerrado con la X.")
        return True
    except Exception as e:
        log(f"No apareció el popup de anuncios (o ya estaba cerrado): {e}")
        return False
    

def leer_config_api(archivo='config.ini', seccion='football_data'):
    ruta_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), archivo)
    parser = ConfigParser()
    parser.read(ruta_config)
    if parser.has_section(seccion):
        return {param[0]: param[1] for param in parser.items(seccion)}
    else:
        raise Exception(f'Sección {seccion} no encontrada en el archivo {archivo}')

def get_api_headers():
    cfg = leer_config_api()
    return {"X-Auth-Token": cfg["api_key"]}

def formato_temporada(api_season):
    return f"{str(api_season)[-2:]}/{str(api_season+1)[-2:]}"

def _read_backup_config():
    """
    Lee config.ini:
      - credenciales en [postgresql] (las que ya usas con psycopg2)
      - sección opcional [Backup] para carpeta/retención/pg_dump_path
    """
    log("🔧 _read_backup_config: inicio")

    # Credenciales como las usa tu utils.conexion_db()
    db_cfg = leer_config_db()  # dict: host, port, database/dbname, user, password
    log(f"🔧 _read_backup_config: credenciales leídas (keys={list(db_cfg.keys())})")

    # Sección [Backup] es opcional
    parser = ConfigParser()
    parser.read("config.ini", encoding="utf-8")

    dest_dir = "./backups"
    retention_days = 30
    pg_dump_path = ""

    if parser.has_section("Backup"):
        dest_dir = parser.get("Backup", "dir", fallback=dest_dir)
        retention_days = parser.getint("Backup", "retention_days", fallback=retention_days)
        pg_dump_path = parser.get("Backup", "pg_dump_path", fallback=pg_dump_path).strip()
        log(f"🔧 _read_backup_config: Backup.dir={dest_dir} · retention_days={retention_days} · pg_dump_path='{pg_dump_path or '(PATH)'}'")
    else:
        log("ℹ️ _read_backup_config: sección [Backup] no encontrada; usando valores por defecto.")

    cfg = {
        "db": db_cfg,
        "dest_dir": dest_dir,
        "retention_days": retention_days,
        "pg_dump_path": pg_dump_path,
    }
    log("✅ _read_backup_config: fin")
    return cfg


def _find_pg_dump(pg_dump_path_hint: str) -> str:
    """
    Localiza pg_dump:
      1) Si viene ruta en config y existe → úsala
      2) PATH del sistema (shutil.which)
      3) Windows: prueba rutas típicas en Program Files
    """
    log("🔎 _find_pg_dump: inicio")

    # 1) pista directa
    if pg_dump_path_hint:
        log(f"🔎 _find_pg_dump: probando pista directa '{pg_dump_path_hint}'")
        if os.path.isfile(pg_dump_path_hint):
            log(f"✅ _find_pg_dump: encontrado fichero '{pg_dump_path_hint}'")
            return pg_dump_path_hint
        if os.path.isdir(pg_dump_path_hint):
            candidate = os.path.join(pg_dump_path_hint, "pg_dump.exe" if os.name == "nt" else "pg_dump")
            if os.path.isfile(candidate):
                log(f"✅ _find_pg_dump: encontrado en carpeta pista → '{candidate}'")
                return candidate
            else:
                log(f"⚠️ _find_pg_dump: no existe '{candidate}' dentro de la carpeta pista")

    # 2) PATH
    exe = "pg_dump.exe" if os.name == "nt" else "pg_dump"
    found = shutil.which(exe)
    if found:
        log(f"✅ _find_pg_dump: encontrado en PATH → '{found}'")
        return found
    else:
        log("ℹ️ _find_pg_dump: no está en PATH")

    # 3) Windows: rutas típicas
    if os.name == "nt":
        bases = [r"C:\Program Files\PostgreSQL", r"C:\Program Files (x86)\PostgreSQL"]
        for base in bases:
            for ver in ("16", "15", "14", "13", "12"):
                cand = os.path.join(base, ver, "bin", "pg_dump.exe")
                if os.path.isfile(cand):
                    log(f"✅ _find_pg_dump: encontrado en ruta típica → '{cand}'")
                    return cand
        log("⚠️ _find_pg_dump: no se encontró en rutas típicas de Windows")

    msg = "No se encontró 'pg_dump'. Añade su ruta en [Backup] pg_dump_path de config.ini o agrégalo al PATH."
    log(f"❌ _find_pg_dump: {msg}")
    raise FileNotFoundError(msg)


def _ensure_dir(path: str):
    log(f"📁 _ensure_dir: asegurando carpeta '{path}'")
    os.makedirs(path, exist_ok=True)


def _rotate_backups(dest_dir: str, prefix: str, retention_days: int):
    """
    Borra ficheros con el prefijo que sean más antiguos que N días.
    """
    log(f"🧹 _rotate_backups: inicio (dir='{dest_dir}', prefix='{prefix}', retention_days={retention_days})")
    cutoff = datetime.now() - timedelta(days=retention_days)
    patron = os.path.join(dest_dir, f"{prefix}_*.sql.gz")
    eliminados = 0

    for fpath in glob.glob(patron):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
            if mtime < cutoff:
                os.remove(fpath)
                eliminados += 1
                log(f"🗑️  _rotate_backups: eliminado backup antiguo → {os.path.basename(fpath)} (mtime={mtime})")
        except Exception as e:
            log(f"⚠️ _rotate_backups: no se pudo eliminar {fpath}: {e}")

    log(f"✅ _rotate_backups: fin (eliminados={eliminados})")


def _run_pg_dump(pg_dump, host, port, dbname, user, password, out_gz_path):
    """
    Ejecuta pg_dump (formato 'plain') y comprime stdout a GZip.
    Lee stderr en un hilo aparte para evitar deadlocks cuando -v genera mucho output.
    """
    cmd = [
        pg_dump,
        "-h", host,
        "-p", str(port),
        "-U", user,
        "-d", dbname,
        "-v",          # verbose (va a stderr)
        "-w",          # no prompt password (usamos env var)
        "-F", "p",     # 'plain'
    ]

    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = password

    log("▶️ _run_pg_dump: ejecutando → " + " ".join(shlex.quote(x) for x in cmd))
    t0 = time.time()

    # Para capturar y evitar bloqueo, leeremos stderr en otro hilo.
    stderr_lines = []
    def _drain_stderr(stream):
        try:
            # texto y lectura línea a línea para log progresivo
            for line in iter(stream.readline, ''):
                line = line.rstrip()
                if line:
                    stderr_lines.append(line)
                    # log opcional por cada objeto (reduce si es muy ruidoso)
                    if ("dumping" in line.lower()) or ("saving" in line.lower()):
                        log(f"pg_dump: {line}")
        except Exception as e:
            log(f"⚠️ _run_pg_dump: error leyendo stderr: {e}")

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,          # streams en texto
        bufsize=1           # line-buffered
    ) as proc, gzip.open(out_gz_path, "wb") as gz_out:

        # Arranca hilo para drenar stderr
        assert proc.stderr is not None
        t_err = threading.Thread(target=_drain_stderr, args=(proc.stderr,), daemon=True)
        t_err.start()

        # Lee stdout en binario (usamos el buffer de texto; reabrimos stream binario)
        total_bytes = 0
        last_log = t0
        try:
            # aunque text=True, podemos leer bytes desde el descriptor del proceso:
            raw_stdout = proc.stdout
            assert raw_stdout is not None

            # En text=True readline es texto; aquí queremos chunks eficientemente:
            # usamos el buffer subyacente para obtener bytes si está disponible
            # Si no, simplemente iteramos por bloques del archivo de texto y re-encode
            while True:
                chunk = raw_stdout.buffer.read(1024 * 256) if hasattr(raw_stdout, "buffer") else raw_stdout.read(1024 * 256)
                if not chunk:
                    break
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8", errors="ignore")
                gz_out.write(chunk)
                total_bytes += len(chunk)

                # Log progresivo cada ~5s
                now = time.time()
                if now - last_log >= 5:
                    log(f"⏳ _run_pg_dump: {total_bytes/1_048_576:0.1f} MiB leídos…")
                    last_log = now

        except Exception as e:
            log(f"❌ _run_pg_dump: excepción durante el volcado: {e}. Matando proceso…")
            proc.kill()
            t_err.join(timeout=1)
            raise e

        # Espera a que termine el proceso y el hilo de stderr
        ret = proc.wait()
        t_err.join(timeout=5)
        elapsed = time.time() - t0

        if ret != 0:
            err_txt = out_gz_path.replace(".sql.gz", ".stderr.txt")
            try:
                with open(err_txt, "w", encoding="utf-8") as f:
                    f.write("\n".join(stderr_lines))
                log(f"❌ _run_pg_dump: pg_dump returncode={ret}. STDERR → {err_txt}")
            except Exception as e:
                log(f"⚠️ _run_pg_dump: no se pudo guardar STDERR: {e}")
            raise RuntimeError(f"pg_dump salió con código {ret}. Revisa {err_txt}.")

    # Métricas finales
    try:
        size_gz = os.path.getsize(out_gz_path)
        log(f"✅ _run_pg_dump: OK en {elapsed:0.1f}s · SQL sin comprimir≈{total_bytes/1_048_576:0.1f} MiB · archivo='{out_gz_path}' ({size_gz/1_048_576:0.1f} MiB gz)")
    except Exception as e:
        log(f"⚠️ _run_pg_dump: OK pero no se pudo obtener tamaño del archivo: {e}")

def _read_cleanup_config():
    parser = ConfigParser()
    parser.read("config.ini", encoding="utf-8")

    # días (por defecto 30)
    days = 30
    if parser.has_section("Cleanup"):
        days = parser.getint("Cleanup", "days", fallback=days)

    # logs_dir
    logs_dir = "./log"
    if parser.has_section("Cleanup"):
        logs_dir = parser.get("Cleanup", "logs_dir", fallback=logs_dir)

    # ok/ko/data csv
    ok_csv_dir = parser.get("Cleanup", "ok_csv_dir", fallback="./ok/csv") if parser.has_section("Cleanup") else "./ok/csv"
    ko_csv_dir = parser.get("Cleanup", "ko_csv_dir", fallback="./ko/csv") if parser.has_section("Cleanup") else "./ko/csv"
    data_csv_dir = parser.get("Cleanup", "data_csv_dir", fallback="./data/csv") if parser.has_section("Cleanup") else "./data/csv"

    # base_csv (el que ya usas para guardar tus outputs)
    try:
        base_csv = get_base_path_from_ini()
    except Exception:
        base_csv = None

    # backups dir desde [Backup]
    backups_dir = None
    if parser.has_section("Backup"):
        backups_dir = parser.get("Backup", "dir", fallback="").strip() or None

    # rutas extra
    extra_dirs = []
    if parser.has_section("Cleanup"):
        extra_raw = parser.get("Cleanup", "extra_dirs", fallback="").strip()
        if extra_raw:
            extra_dirs = [s.strip() for s in extra_raw.split(";") if s.strip()]

    # set final (evita duplicados y None)
    candidates = {
        logs_dir,
        ok_csv_dir,
        ko_csv_dir,
        data_csv_dir,
        "./data",       # también limpia ficheros sueltos aquí
        base_csv or "",
        backups_dir or "",
    }
    # añade extras
    candidates.update(extra_dirs)
    # filtra vacíos
    target_dirs = [d for d in candidates if d]

    return {
        "days": days,
        "dirs": target_dirs,
    }


def _is_file_older_than(path, cutoff_dt: datetime) -> bool:
    try:
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime) < cutoff_dt
    except Exception:
        return False


def _iter_files_in_dir(root_dir: str):
    """
    Itera SOLO ficheros (no carpetas) dentro de root_dir y subcarpetas.
    """
    # Usa glob recursivo
    pattern = os.path.join(root_dir, "**", "*")
    for f in glob.iglob(pattern, recursive=True):
        if os.path.isfile(f):
            yield f


def _safe_to_delete(path: str) -> bool:
    """
    Reglas de seguridad mínimas:
     - No borrar archivos ocultos del sistema (.git, .svn) ni .gitkeep
     - No borrar ficheros .stderr.txt generados por procesos en curso en los últimos 10 minutos
       (por si están siendo escritos ahora mismo)
    """
    name = os.path.basename(path).lower()
    if name in (".gitkeep",):
        return False
    if name.startswith(".git") or name.startswith(".svn"):
        return False
    # evita borrar .stderr.txt demasiado recientes
    if name.endswith(".stderr.txt"):
        try:
            if (time.time() - os.path.getmtime(path)) < 600:  # 10 minutos
                return False
        except Exception:
            pass
    return True


def _delete_old_files_in_dir(root_dir: str, days: int) -> dict:
    """
    Borra ficheros con mtime < hoy - days.
    Devuelve dict resumen con contadores.
    """
    summary = {
        "dir": root_dir,
        "checked": 0,
        "deleted": 0,
        "errors": 0,
    }
    if not os.path.isdir(root_dir):
        log(f"ℹ️ cleanup: directorio no existe, se omite → {root_dir}")
        return summary

    cutoff_dt = datetime.now() - timedelta(days=days)
    log(f"🧹 cleanup: procesando '{root_dir}' (cutoff={cutoff_dt:%Y-%m-%d %H:%M:%S}, days={days})")

    for f in _iter_files_in_dir(root_dir):
        summary["checked"] += 1
        if not _safe_to_delete(f):
            continue
        try:
            if _is_file_older_than(f, cutoff_dt):
                try:
                    os.remove(f)
                    summary["deleted"] += 1
                    log(f"🗑️  cleanup: eliminado → {f}")
                except Exception as e:
                    summary["errors"] += 1
                    log(f"⚠️ cleanup: error eliminando '{f}': {e}")
        except Exception as e:
            summary["errors"] += 1
            log(f"⚠️ cleanup: error revisando '{f}': {e}")

    log(f"✅ cleanup resumen '{root_dir}': revisados={summary['checked']} · borrados={summary['deleted']} · errores={summary['errors']}")
    return summary
