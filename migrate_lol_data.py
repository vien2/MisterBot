import psycopg2
import psycopg2.extras
import os
from datetime import datetime

# CONFIGURACIÓN (Updated)
OLD_DB_CONN = "dbname='LoL_Stats' user='vien2' host=''"
NEW_DB_CONN = "dbname='misterbot' user='vien2' host=''" 

def get_connection(conn_str):
    try:
        conn = psycopg2.connect(conn_str)
        return conn
    except Exception as e:
        print(f"Error conectando a {conn_str}: {e}")
        return None

def migrate_table(old_conn, new_conn, table_name, columns_map, pk_columns):
    print(f"--- Migrando tabla: {table_name} ---")
    
    try:
        with old_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur_old:
            cur_old.execute(f"SELECT * FROM public.{table_name}")
            rows = cur_old.fetchall()
            
            if not rows:
                print(f"  Tabla {table_name} vacía.")
                return

            print(f"  Leídos {len(rows)} registros. Insertando...")
            
            with new_conn.cursor() as cur_new:
                count = 0
                skipped = 0
                for row in rows:
                    values = {}
                    for old_col, new_col in columns_map.items():
                        if old_col in row:
                            values[new_col] = row[old_col]
                    
                    # VALIDACIONES ESPECÍFICAS
                    if table_name == 'team_game_stats':
                        if not values.get('team_id'): # Skip null or 0
                            skipped += 1
                            continue
                            
                    if table_name == 'game_stats':
                        if not values.get('player_id'): # Skip null or 0
                            skipped += 1
                            continue
                    
                    # Añadir campos extra SOLO si no es scrape_logs (que no los tiene)
                    if table_name != 'scrape_logs':
                        values['f_carga'] = datetime.now()
                        values['hash'] = ''
                    
                    cols = ', '.join(f'"{k}"' for k in values.keys())
                    vals_placeholders = ', '.join(['%s'] * len(values))
                    conflict_target = ', '.join(pk_columns)
                    
                    # Scrape Logs usa ID serial como PK, usamos conflicto en entity si existe, o nada
                    if table_name == 'scrape_logs':
                         # Ojo: scrape_logs vieja tiene ID, la nueva es serial.
                         # Si mapeamos ID -> ID, puede haber conflicto con secuencia.
                         # Mejor NO insertar ID y dejar que la nueva genere uno,
                         # PERO la tabla vieja tiene el historial.
                         # Para evitar líos: INSERT sin ID (que genere nuevos serials)
                         if 'id' in values: del values['id']
                         # Quitamos columnas conflictivas del insert statement
                         cols = ', '.join(f'"{k}"' for k in values.keys())
                         vals_placeholders = ', '.join(['%s'] * len(values))
                         # INSERT simple (sin ON CONFLICT porque ID es nuevo)
                         query = f"""
                            INSERT INTO "LoL_Stats".{table_name} ({cols}) 
                            VALUES ({vals_placeholders})
                         """
                    else:
                        query = f"""
                            INSERT INTO "LoL_Stats".{table_name} ({cols}) 
                            VALUES ({vals_placeholders})
                            ON CONFLICT ({conflict_target}) DO NOTHING
                        """
                    
                    cur_new.execute(query, list(values.values()))
                    count += 1
                
                new_conn.commit()
                print(f"  Migrados: {count}. Omitidos (Null IDs): {skipped}.")

    except Exception as e:
        new_conn.rollback()
        print(f"  ERROR migrando {table_name}: {e}")

def main():
    print("Iniciando migración (V2 - Robust)...")
    old_conn = get_connection(OLD_DB_CONN)
    new_conn = get_connection(NEW_DB_CONN)
    
    if not old_conn or not new_conn:
        print("No se pudo conectar.")
        return

    # 1. TORNEOS
    migrate_table(old_conn, new_conn, "tournaments", {
        "id": "id", "name": "name", "slug": "slug", 
        "start_date": "start_date", "end_date": "end_date",
        "season": "season", "region": "region"
    }, ["id"])

    # 2. TEAMS
    migrate_table(old_conn, new_conn, "teams", {
        "id": "id", "name": "name", "code": "code",
        "image_url": "image_url", "region": "region"
    }, ["id"])

    # 3. PLAYERS
    migrate_table(old_conn, new_conn, "players", {
        "id": "id", "handle": "handle", "real_name": "real_name",
        "role": "role", "team_id": "team_id", "image_url": "image_url"
    }, ["id"])
    
    # 4. MATCHES
    migrate_table(old_conn, new_conn, "matches", {
        "id": "id", "tournament_id": "tournament_id", "match_date": "match_date",
        "patch": "patch", "week": "week", "best_of": "best_of",
        "game_number": "game_number", "blue_team_id": "blue_team_id",
        "red_team_id": "red_team_id", "winner_id": "winner_id", "score": "score"
    }, ["id"])

    # 5. TEAM GAME STATS
    cols_team = [
        "match_id", "team_id", "side", "win", "game_duration", "patch", "bans", "picks",
        "total_kills", "total_gold", "towers_destroyed", "dragons_killed", "barons_killed",
        "void_grubs", "rift_heralds", "plates_total", "plates_top", "plates_mid", "plates_bot",
        "wards_destroyed", "wards_placed", "jungle_share_15", "jungle_share_end",
        "first_blood", "first_tower", "first_blood_time", "first_tower_time",
        "first_dragon_time", "first_baron_time", "dragon_events", "elder_dragons_killed",
        "total_deaths", "total_assists"
    ]
    map_team = {c: c for c in cols_team}
    migrate_table(old_conn, new_conn, "team_game_stats", map_team, ["match_id", "team_id"])

    # 6. GAME STATS
    cols_game = [
        "match_id", "player_id", "champion_name", "side", "win", "kills", "deaths", "assists",
        "total_gold", "cs", "level", "vision_score", "wards_placed", "wards_killed",
        "control_wards_purchased", "detector_wards_placed", "damage_dealt",
        "physical_damage_dealt_to_champions", "magic_damage_dealt_to_champions",
        "true_damage_dealt_to_champions", "damage_taken", "damage_self_mitigated",
        "damage_dealt_to_turrets", "gold_per_min", "gold_share", "cs_per_min",
        "gold_diff_15", "cs_diff_15", "xp_diff_15", "level_diff_15", "kill_participation",
        "solo_kills", "time_ccing_others", "total_time_cc_dealt", "total_heal",
        "total_heals_on_teammates", "damage_share", "vision_share", "vision_score_per_minute",
        "wards_per_minute", "wards_cleared_per_minute", "cs_in_team_jungle", "cs_in_enemy_jungle",
        "consumables_purchased", "items_purchased", "shutdown_bounty_collected",
        "shutdown_bounty_lost", "objectives_stolen", "double_kills", "triple_kills",
        "quadra_kills", "penta_kills", "total_damage_shielded_on_teammates", "total_time_spent_dead"
    ]
    map_game = {c: c for c in cols_game}
    migrate_table(old_conn, new_conn, "game_stats", map_game, ["match_id", "player_id"])

    # 7. SCRAPE LOGS (Sin f_carga)
    migrate_table(old_conn, new_conn, "scrape_logs", {
        "entity_type": "entity_type", "entity_id": "entity_id",
        "status": "status", "last_scraped": "last_scraped", "details": "details"
    }, ["id"])

    print("\n--- Migración FINALIZADA ---")
    
    # Ajustar secuencias
    try:
        with new_conn.cursor() as cur:
            cur.execute("""
                SELECT setval('"LoL_Stats".players_id_seq', COALESCE((SELECT MAX(id) FROM "LoL_Stats".players), 1));
                SELECT setval('"LoL_Stats".scrape_logs_id_seq', COALESCE((SELECT MAX(id) FROM "LoL_Stats".scrape_logs), 1));
            """)
            new_conn.commit()
            print("Secuencias actualizadas.")
    except Exception as e:
        print(f"Warning actualizando secuencias: {e}")

    old_conn.close()
    new_conn.close()

if __name__ == "__main__":
    main()
