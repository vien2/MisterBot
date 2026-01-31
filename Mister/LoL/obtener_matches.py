import time
import datetime
import urllib.parse
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils import log, conexion_db
from .utils_lol import log_scrape_status

def obtener_matches(driver, schema="lol_stats", **kwargs):
    """
    Scrapes scan active tournaments to find matches.
    Populates the 'matches' table.
    """
    log("Iniciando Discovery de Partidos (obtener_matches)...")
    data_list = []

    # 1. Obtener Torneos Activos o Recientes de la DB
    tournaments_to_scan = []
    
    # Si pasan argumentos manuales
    if 'target_tournaments' in kwargs:
        tournaments_to_scan = kwargs['target_tournaments'] # Lista de slugs
    else:
        # Buscar en DB torneos activos (endpoint > hoy - 7 dias de margen)
        try:
            with conexion_db() as conn:
                with conn.cursor() as cur:
                    # Ajusta la query según tus necesidades. Aquí pillamos los de 'S16' o activos.
                    cur.execute(f"""
                        SELECT id, slug FROM {schema}.tournaments 
                        WHERE season = 'S16' 
                           OR end_date >= CURRENT_DATE 
                           OR end_date IS NULL
                    """)
                    rows = cur.fetchall()
                    tournaments_to_scan = [r[1] for r in rows]
        except Exception as e:
            log(f"Error recuperando torneos de DB: {e}")
            return []

    log(f"Torneos a escanear: {len(tournaments_to_scan)}")

    for t_idx, tournament_slug in enumerate(tournaments_to_scan):
        log(f"[{t_idx+1}/{len(tournaments_to_scan)}] Escaneando torneo: {tournament_slug}")
        
        try:
            # URL de lista de partidos del torneo
            url = f"https://gol.gg/tournament/tournament-matchlist/{tournament_slug}/"
            driver.get(url)
            
            try:
                wait = WebDriverWait(driver, 10)
                # Esperar a la tabla. A veces no hay tabla si es muy nuevo.
                table = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "table_list")))
            except:
                log(f"    No se encontró tabla de partidos para {tournament_slug}")
                continue

            rows = table.find_elements(By.TAG_NAME, "tr")
            # Skip header
            for row in rows[1:]:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) < 4: continue
                
                # Columnas usuales: Name(Link), Date, Region, Patch...
                # La estructura cambia a veces.
                # Col 0: Name (Link to match -> ID)
                # Col 1: Date
                
                try:
                    link_el = cols[0].find_element(By.TAG_NAME, "a")
                    match_href = link_el.get_attribute("href")
                    match_name = link_el.text.strip() # "T1 vs GEN"
                except:
                    continue

                # Extract ID from href: ../game/stats/57366/page-game/
                # match_href suele ser ../../game/stats/ID/page-game/
                parts = match_href.split("/")
                match_id = None
                for p in parts:
                    if p.isdigit(): 
                        match_id = p
                        break
                
                if not match_id: continue

                # Extract Date
                date_str = cols[1].text.strip()
                # Default to a valid dummy date to avoid NaN/Float errors in Pandas/Postgres
                match_date = "1900-01-01" 
                if date_str and date_str != "-":
                    try:
                        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                        match_date = dt.strftime("%Y-%m-%d")
                    except:
                        pass # Keeps 1900-01-01

                # Extract Patch
                patch_str = ""
                if len(cols) > 5:
                    patch_str = cols[5].text.strip()
                
                # Teams Parsing (From "T1 vs GEN")
                blue_team_id = None
                red_team_id = None
                # Esto es complejo sin tener los IDs de equipos a mano.
                # Para la tabla 'matches', a menudo solo guardamos el texto o intentamos resolver.
                # Por simplicidad en ESTA fase, guardaremos lo básico.
                # El scraper de DETALLE (team_game_stats) es quien confirma los IDs de equipos reales.
                
                row_dict = {
                    "id": match_id,
                    "tournament_id": tournament_slug, # Ojo: FK a tournament.id
                    "match_date": match_date,
                    "patch": patch_str,
                    "game_number": 1, # Default, difícil saber si es Bo3 game 1 o 2 desde aquí a veces
                    # Los demás campos se pueden llenar luego o dejar NULL
                }
                
                # Parsear resultado (WIN/LOSS) o Score si está disponible
                # En la lista a veces sale "1 - 0".
                
                data_list.append(row_dict)
        
        except Exception as e:
            log(f"Error escaneando torneo {tournament_slug}: {e}")

    log(f"Discovery completado. Partidos encontrados: {len(data_list)}")
    return data_list
