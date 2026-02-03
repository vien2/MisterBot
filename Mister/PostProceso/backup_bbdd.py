# -*- coding: utf-8 -*-
"""
Postproceso: backup de la base de datos PostgreSQL
 - Usa pg_dump y comprime a .sql.gz
 - Descubre pg_dump (PATH o config)
 - Mantiene una política de retención (borra backups antiguos)
"""

import os
import time
from datetime import datetime
from utils import log,_read_backup_config,_find_pg_dump,_find_pg_dumpall,_ensure_dir,_run_pg_dump,_run_pg_dumpall,_rotate_backups


def backup_bbdd(conn, schema=None):
    """
    Firma compatible con tu orquestador para postprocesos tipo 'psql':
      - Recibe 'conn' (psycopg2), que aquí NO usamos.
      - 'schema' es ignorado: el backup es de toda la BD.
    """
    _ = (conn, schema)
    log("backup_bbdd: inicio")

    cfg = _read_backup_config()

    db = cfg["db"]
    host = db.get("host", "localhost")
    port = db.get("port", "5432")
    dbname = db.get("database") or db.get("dbname") or db.get("db")  # según cómo esté en tu config.ini
    user = db.get("user") or db.get("username")
    password = db.get("password", "")

    if not dbname or not user:
        raise RuntimeError("Config [postgresql] debe incluir 'database/dbname' y 'user'.")

    dest_dir = cfg["dest_dir"]
    retention_days = int(cfg["retention_days"])
    backup_all = cfg.get("backup_all", False)

    _ensure_dir(dest_dir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if backup_all:
        prefix = "backup_ALL_DBS"
        pg_tool = _find_pg_dumpall(cfg["pg_dump_path"])
    else:
        prefix = f"backup_{dbname}"
        pg_tool = _find_pg_dump(cfg["pg_dump_path"])

    out_name = f"{prefix}_{ts}.sql.gz"
    out_path = os.path.join(dest_dir, out_name)

    log(f"Destino: {out_path} (Backup all: {backup_all})")

    try:
        t0 = time.time()
        
        if backup_all:
             _run_pg_dumpall(
                pg_dumpall=pg_tool,
                host=host,
                port=port,
                user=user,
                password=password,
                out_gz_path=out_path
            )
        else:
            _run_pg_dump(
                pg_dump=pg_tool,
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password,
                out_gz_path=out_path
            )

        secs = time.time() - t0
        size_mb = os.path.getsize(out_path) / (1024 * 1024.0)
        log(f"✅ Backup OK en {secs:0.1f}s · {size_mb:0.1f} MB → {out_path}")

        _rotate_backups(dest_dir, prefix, retention_days)
        log("backup_bbdd: fin")
        return out_path

    except Exception as e:
        log(f"❌ Error en backup: {e}")
        raise

if __name__ == "__main__":
    backup_bbdd(None)
