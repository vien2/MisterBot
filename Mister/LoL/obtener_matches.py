import time
import re
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
            # 1. Collect all match links and metadata first to avoid StaleElementReference
            match_links = []
            for row in rows[1:]:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) < 4: continue
                try:
                    link_el = cols[0].find_element(By.TAG_NAME, "a")
                    match_href = link_el.get_attribute("href")
                    match_name = link_el.text.strip()
                    
                    # Date & Patch from list (fallbacks)
                    date_str = cols[1].text.strip()
                    match_date = "1900-01-01"
                    if date_str and date_str != "-":
                        try:
                            # Intentar formatear. Gol.gg suele usar YYYY-MM-DD
                            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                            match_date = dt.strftime("%Y-%m-%d")
                        except ValueError:
                             # log(f"⚠️ Formato de fecha desconocido: '{date_str}'")
                             pass
                        except Exception: pass
                    
                    patch_str = ""
                    if len(cols) > 5:
                        patch_str = cols[5].text.strip()

                    # FILTER: Check Score (Column 2)
                    # If it contains "vs" or no score, it's unplayed.
                    score_str = cols[2].text.strip() # Keep original case for saving
                    score_lower = score_str.lower()
                    
                    # Si filtramos los no jugados, activamos esto:
                    if "vs" in score_lower or "-" not in score_lower:
                        # log(f"    Saltando partido no jugado: {match_name} ({date_str})")
                        continue

                    match_links.append({
                        "href": match_href,
                        "name": match_name,
                        "date": match_date,
                        "patch": patch_str,
                        "score": score_str
                    })
                except: continue

            log(f"    Encontradas {len(match_links)} series posibles. Entrando a verificar...")

            # 2. Visit each match page
            for m_meta in match_links:
                match_href = m_meta["href"]
                match_name = m_meta["name"]
                match_date = m_meta["date"]
                patch_str = m_meta["patch"]
                match_score = m_meta.get("score", "")

                # Expansion of Series (For Bo3/Bo5)
                try:
                    driver.get(match_href)
                    time.sleep(1) 

                    game_urls_dict = {} # {game_num: url}
                    
                    # 1. Buscar todos los botones del menú de juegos (activos e inactivos)
                    # El selector li[class*='game-menu-button'] pilla tanto 'game-menu-button' como 'game-menu-button-active'
                    menu_items = driver.find_elements(By.CSS_SELECTOR, "ul.navbar-nav li[class*='game-menu-button']")
                    
                    for item in menu_items:
                        txt = item.get_attribute("textContent").strip()
                        if "Game" in txt:
                            try:
                                num_match = re.search(r"\d+", txt)
                                if not num_match: continue
                                num = int(num_match.group())
                                
                                # Buscar link dentro del LI
                                links = item.find_elements(By.TAG_NAME, "a")
                                if links:
                                    g_url = links[0].get_attribute("href")
                                else:
                                    # Si no hay link, es que ya estamos en esta página (es el activo)
                                    g_url = driver.current_url
                                
                                game_urls_dict[num] = g_url
                            except: pass
                    
                    # 2. Si no encontramos nada arriba, es un Bo1 (o el menú es distinto)
                    if not game_urls_dict:
                        game_urls_dict[1] = match_href
                    
                    # 3. Convertir a lista y ordenar
                    game_urls = sorted(game_urls_dict.items()) # [(1, url), (2, url)...]
                    
                    # Determinar Best of (si hay 4 o 5 juegos, seguro Bo5. Si hay 1-3, depende)
                    # Por defecto ponemos 3 si hay > 1, 5 si hay > 3. 
                    max_g = max(game_urls_dict.keys()) if game_urls_dict else 1
                    best_of = 1
                    if max_g > 1:
                        best_of = 3 if max_g <= 3 else 5
                        # Tip: Si el nombre del torneo tiene "Playoffs" o "Knockout", suele ser Bo5
                        if any(x in tournament_slug.lower() for x in ["playoff", "knockout", "final"]):
                             best_of = 5

                    series_id = None
                    if game_urls:
                        try:
                            # El ID de la serie suele ser el ID del primer juego
                            series_id = re.search(r"stats/(\d+)/", game_urls[0][1]).group(1)
                        except: pass

                    for g_num, g_url in game_urls:
                        try:
                            g_id = re.search(r"stats/(\d+)/", g_url).group(1)
                            if any(d['id'] == g_id for d in data_list): continue

                            data_list.append({
                                "id": g_id,
                                "tournament_id": tournament_slug,
                                "match_date": match_date,
                                "patch": patch_str,
                                "game_number": g_num,
                                "series_id": series_id,
                                "best_of": best_of,
                                "score": match_score
                            })
                        except: continue
                    
                except Exception as e_inner:
                    log(f"    Error expandiendo serie {match_name}: {e_inner}")

        
        except Exception as e:
            log(f"Error escaneando torneo {tournament_slug}: {e}")

    log(f"Discovery completado. Partidos encontrados: {len(data_list)}")
    return data_list
