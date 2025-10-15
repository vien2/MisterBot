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
