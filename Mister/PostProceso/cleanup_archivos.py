# -*- coding: utf-8 -*-
"""
Postproceso: limpieza de archivos antiguos en varias rutas.
Borra ficheros con mtime < hoy - N días (por defecto 30).
Rutas por defecto:
 - Backups.dir (si existe en [Backup])
 - paths.base_csv (si existe en [paths])
 - ./ok/csv
 - ./ko/csv
 - ./log
 - ./data/csv
 - ./data  (solo ficheros sueltos)
Se pueden añadir rutas extra en [Cleanup] extra_dirs (separadas por ';').

Config opcional en config.ini:
[Cleanup]
days=30
logs_dir=./log
ok_csv_dir=./ok/csv
ko_csv_dir=./ko/csv
data_csv_dir=./data/csv
extra_dirs=C:\otra\carpeta;./otra_relativa

[Backup]
dir=... (ya lo usas)

[paths]
base_csv=... (ya lo usas)
"""
from utils import log,_read_cleanup_config,_delete_old_files_in_dir

def cleanup_archivos(conn, schema=None):
    """
    Postproceso tipo 'psql' (recibe 'conn', no se usa). Limpia archivos antiguos.
    """
    _ = (conn, schema)
    log("cleanup_archivos: inicio")

    cfg = _read_cleanup_config()
    days = int(cfg["days"])
    dirs = cfg["dirs"]

    total_checked = total_deleted = total_errors = 0
    for d in dirs:
        s = _delete_old_files_in_dir(d, days)
        total_checked += s["checked"]
        total_deleted += s["deleted"]
        total_errors  += s["errors"]

    log(f"cleanup_archivos: fin · revisados={total_checked} · borrados={total_deleted} · errores={total_errors}")
    return {
        "checked": total_checked,
        "deleted": total_deleted,
        "errors": total_errors
    }
