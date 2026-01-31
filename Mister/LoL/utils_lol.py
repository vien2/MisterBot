import psycopg2
from utils import log, conexion_db

def get_completed_matches(schema="LoL_Stats", entity_type="MATCH"):
    """
    Devuelve un set con los IDs de los matches ya scrapeados (COMPLETED).
    """
    completed_ids = set()
    try:
        with conexion_db() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT entity_id FROM {schema}.scrape_logs WHERE entity_type = %s AND status = 'COMPLETED'", (entity_type,))
                rows = cur.fetchall()
                completed_ids = {row[0] for row in rows}
    except Exception as e:
        log(f"Error recuperando matches completados: {e}")
    return completed_ids

def get_match_ids(target_ids=None, schema="LoL_Stats"):
    """
    Devuelve lista de objetos sim-dict con 'id', 'blue_team_id', 'red_team_id'
    para los matches a procesar.
    """
    matches = []
    try:
        with conexion_db() as conn:
            with conn.cursor() as cur:
                if target_ids:
                    # Traer específicos
                    placeholders = ','.join(['%s'] * len(target_ids))
                    query = f"SELECT id, blue_team_id, red_team_id, winner_id FROM {schema}.matches WHERE id IN ({placeholders})"
                    cur.execute(query, tuple(target_ids))
                else:
                    # Traer todos
                    query = f"SELECT id, blue_team_id, red_team_id, winner_id FROM {schema}.matches"
                    cur.execute(query)
                
                rows = cur.fetchall()
                # Convert to simple objects
                for r in rows:
                    class MatchObj: pass
                    m = MatchObj()
                    m.id = r[0]
                    m.blue_team_id = r[1]
                    m.red_team_id = r[2]
                    m.winner_id = r[3]
                    matches.append(m)
    except Exception as e:
        log(f"Error recuperando IDs de matches: {e}")
    return matches

def obtener_o_crear_jugador(handle, team_id, role, real_name=None, schema="LoL_Stats"):
    """
    Busca jugador por handle (case insensitive). Si no existe, lo crea.
    Devuelve el player_id.
    """
    player_id = None
    try:
        with conexion_db() as conn:
            with conn.cursor() as cur:
                # 1. Buscar
                cur.execute(f"SELECT id FROM {schema}.players WHERE LOWER(handle) = LOWER(%s)", (handle,))
                res = cur.fetchone()
                if res:
                    return res[0]
                
                # 2. Insertar si no existe
                log(f"[INFO] Creando jugador: '{handle}' (Team: {team_id}, Role: {role})")
                insert_query = f"""
                    INSERT INTO {schema}.players (handle, real_name, role, team_id)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """
                cur.execute(insert_query, (handle, real_name or handle, role, team_id))
                player_id = cur.fetchone()[0]
            conn.commit() # Importante commit para que persista
    except Exception as e:
        log(f"Error gestionando jugador {handle}: {e}")
    
    return player_id

def log_scrape_status(entity_id, entity_type, status, details=None, schema="LoL_Stats"):
    """
    Registra el estado del scrape en scrape_logs.
    """
    try:
        with conexion_db() as conn:
            with conn.cursor() as cur:
                # Upsert log (Delete old, insert new logic or update)
                # Simple approach: Check exist, Update or Insert
                cur.execute(f"SELECT id FROM {schema}.scrape_logs WHERE entity_id = %s AND entity_type = %s", (entity_id, entity_type))
                existing = cur.fetchone()
                
                if existing:
                    cur.execute(f"""
                        UPDATE {schema}.scrape_logs 
                        SET status = %s, last_scraped = NOW(), details = %s
                        WHERE id = %s
                    """, (status, details, existing[0]))
                else:
                    cur.execute(f"""
                        INSERT INTO {schema}.scrape_logs (entity_type, entity_id, status, last_scraped, details)
                        VALUES (%s, %s, %s, NOW(), %s)
                    """, (entity_type, entity_id, status, details))
            conn.commit()
    except Exception as e:
        log(f"Error logging scrape status: {e}")
