#!/opt/misterbot/venv/bin/python
import os, sys, subprocess, shlex, time
import argparse
from contextlib import contextmanager

# CONFIG
DB_DSN = "postgresql://vien2@localhost:5432/misterbot"
PROJECT_DIR = "/home/vien2/MisterBot/Mister"
VENV_PY = "/opt/misterbot/venv/bin/python"
LOCK_FILE_BASE = "/var/lock/misterbot-batch"

try:
    import psycopg
except ImportError:
    print("[ERROR] Falta psycopg. Instala en el venv: pip install psycopg[binary]", flush=True)
    sys.exit(1)

@contextmanager
def file_lock(path):
    import fcntl
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    except BlockingIOError:
        print(f"[WARN] Otro batch ({path}) sigue en ejecución. Salgo.", flush=True)
        sys.exit(0)
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        except Exception:
            pass

def fetch_schedule(batch_group):
    # Filtramos por columna batch_group (por defecto 'Mister' si es NULL)
    sql = """
        SELECT idload, COALESCE(args, '') AS args
        FROM dbo.load_schedule
        WHERE enabled = true
          AND COALESCE(batch_group, 'Mister') = %s
        ORDER BY run_order ASC, idload ASC
    """
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (batch_group,))
            return cur.fetchall()

def run_load(idload: int, args: str):
    cmd = [VENV_PY, "main.py", str(idload)]
    if args.strip():
        cmd += shlex.split(args)
    print(f"[INFO] Ejecutando: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    return subprocess.call(cmd, cwd=PROJECT_DIR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_group", nargs="?", default="Mister", help="Grupo de carga a ejecutar (Mister, tarde, etc)")
    args = parser.parse_args()
    
    group = args.batch_group
    lock_file = f"{LOCK_FILE_BASE}-{group}.lock"

    print(f"== MisterBot batch [{group}] start: {time.strftime('%Y-%m-%dT%H:%M:%S%z')} ==")
    
    with file_lock(lock_file):
        rows = fetch_schedule(group)
        if not rows:
            print(f"[INFO] No hay cargas habilitadas para el grupo '{group}'.")
            return 0

        failures = []
        for idload, load_args in rows:
            rc = run_load(idload, load_args)
            if rc != 0:
                print(f"[ERROR] idload {idload} terminó con código {rc}", flush=True)
                failures.append((idload, rc))

        if failures:
            print("[WARN] Hubo fallos en:", failures, flush=True)
            return 1

    print(f"== MisterBot batch [{group}] end: {time.strftime('%Y-%m-%dT%H:%M:%S%z')} ==")
    return 0

if __name__ == "__main__":
    sys.exit(main())
