import time
import requests
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from utils import log, conexion_db
from .utils_lol import log_scrape_status

def obtener_teams(driver, schema="lol_stats", **kwargs):
    """
    Scrapes active tournaments to find participating TEAMS.
    Populates the 'teams' table.
    Uses Selenium driver passed from main.
    """
    log("Iniciando Discovery de Equipos (obtener_teams)...")
    data_list = []
    seen_ids = set()

    # 1. Obtener Torneos Activos
    tournaments_to_scan = []
    if 'target_tournaments' in kwargs:
        tournaments_to_scan = kwargs['target_tournaments']
    else:
        try:
            with conexion_db() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT slug, region FROM {schema}.tournaments 
                        WHERE season = 'S16' 
                           OR end_date >= CURRENT_DATE 
                           OR end_date IS NULL
                    """)
                    rows = cur.fetchall()
                    tournaments_to_scan = rows # [(slug, region), ...]
        except Exception as e:
            log(f"Error recuperando torneos de DB: {e}")
            return []

    log(f"Torneos a escanear para equipos: {len(tournaments_to_scan)}")

    base_url = "https://gol.gg/tournament/tournament-ranking/{}/" # Usar ranking para ver lista limpia

    for idx, (t_slug, t_region) in enumerate(tournaments_to_scan):
        log(f"[{idx+1}/{len(tournaments_to_scan)}] Escaneando equipos de: {t_slug}")
        
        try:
            url = base_url.format(t_slug)
            driver.get(url)
            time.sleep(1) # Cortesía

            # Buscar tabla de standings/ranking
            # Normalmente es una tabla con clase 'table_list'
            # Row: Pos | Team (Link) | W | L ...
            
            # Buscar filas de equipos (cualquier tabla)
            # Strategy: Find all TRs, check if they have a link to 'team-stats'
            found_teams_in_page = 0
            
            all_rows = driver.find_elements(By.TAG_NAME, "tr")
            
            for row in all_rows:
                try:
                    # Look for Link in this row
                    links = row.find_elements(By.TAG_NAME, "a")
                    target_link = None
                    
                    for lnk in links:
                        h = lnk.get_attribute("href")
                        if h and "team-stats" in h:
                            target_link = lnk
                            break
                    
                    if not target_link:
                        continue
                        
                    # Extract Name & ID
                    name = target_link.text.strip()
                    href = target_link.get_attribute("href")
                    
                    # href: .../team-stats/2804/split...
                    # Split logic adjusted for robustness
                    if "team-stats/" not in href: continue
                    
                    parts = href.split("team-stats/")[1].split("/")
                    team_id = int(parts[0])
                    
                    if team_id in seen_ids: continue
                    
                    # Image extraction (try to find img in same row)
                    img_url = ""
                    try:
                        imgs = row.find_elements(By.TAG_NAME, "img")
                        if imgs: img_url = imgs[0].get_attribute("src")
                    except: pass
                    
                    code = name if len(name) <= 6 else name[:3].upper()
                    
                    data_list.append({
                        "id": team_id,
                        "name": name,
                        "code": code,
                        "image_url": img_url,
                        "region": t_region
                    })
                    seen_ids.add(team_id)
                    found_teams_in_page += 1

                except Exception as row_ex:
                    continue
            
            # log(f"    Equipos en {t_slug}: {found_teams_in_page}")

        except Exception as e:
            log(f"Error procesando torneo {t_slug}: {e}")

    log(f"Equipos encontrados: {len(data_list)}")
    return data_list
