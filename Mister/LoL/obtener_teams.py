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
                    
                    # Fallback: si no hay texto, intentar title o el alt de la imagen
                    if not name:
                         name = target_link.get_attribute("title")
                    
                    # Image extraction (try to find img in same row)
                    img_url = ""
                    try:
                        imgs = row.find_elements(By.TAG_NAME, "img")
                        if imgs: 
                            # Priorizar data-src si existe (lazy load), sino src
                            img_src = imgs[0].get_attribute("data-src") or imgs[0].get_attribute("src")
                            if img_src:
                                img_url = img_src
                                if not name: # Último intento de nombre
                                    name = imgs[0].get_attribute("alt")
                    except: pass
                    
                    # Fallback FINAL: Si no hay imagen, intentar construirla
                    # Gol.gg suele usar: https://gol.gg/_img/teams/Nombre.png
                    if not img_url and name:
                        # OJO: Los nombres pueden tener espacios, hay que ver si la web usa %20 o espacios
                        # Normalmente en HTML src funcionan los espacios si el navegador lo encodeara,
                        # pero mejor dejarlo tal cual string y que el frontend lo maneje, o arriesgarnos.
                        # Probemos con la URL base estándar:
                        img_url = f"https://gol.gg/_img/teams/{name}.png"
                        # log(f"⚠️ Imagen construida manualmente para {name}: {img_url}")

                    # Si sigue sin nombre, saltamos (no podemos guardar un equipo fantasma)
                    if not name:
                        # log(f"Equipo ignorado (ID detectado pero sin nombre): {href}")
                        continue

                    # Limpiar nombre (a veces traen basura del title)
                    # Ejemplo: "Los Ratones stats in LEC 2026 Versus Season" -> "Los Ratones"
                    name = name.strip()
                    if "stats in" in name:
                        name = name.split("stats in")[0].strip()

                    href = target_link.get_attribute("href")
                    
                    # href: .../team-stats/2804/split...
                    # Split logic adjusted for robustness
                    if "team-stats/" not in href: continue
                    
                    parts = href.split("team-stats/")[1].split("/")
                    team_id = int(parts[0])
                    
                    if team_id in seen_ids: continue

                    code = name if len(name) <= 6 else name[:3].upper()
                    
                    # --- INTENTO DE LOCALIZAR LOGO LOCAL (static/logos) ---
                    # Prioridad: 
                    # 1. code.lower().png (ej: fnc.png, th.png, vit.png)
                    # 2. slug_name.webp (ej: team_heretics.webp)
                    
                    import os
                    # Ajusta esta ruta a tu estructura real en el VPS
                    # Asumimos que MisterBot y MisterBot_Web están al mismo nivel en home
                    # /home/vien2/MisterBot/Mister (estamos aqui)
                    # /home/vien2/MisterBot_Web/MisterBot_Web/static/logos (buscamos aqui)
                    
                    # Intento de autodetectar ruta relativa o absoluta común
                    possible_paths = [
                        "/home/vien2/MisterBot_Web/static/logos", # VPS Correcto
                        "/home/vien2/MisterBot_Web/MisterBot_Web/static/logos", # VPS Alternativo
                        "../MisterBot_Web/static/logos", # Local dev relativo
                        "../../MisterBot_Web/MisterBot_Web/static/logos",
                        "C:/Python/MisterBot_Web/MisterBot_Web/static/logos" # Local Windows absolute
                    ]
                    
                    local_logo_path = None
                    final_static_url = ""
                    
                    logo_dir = None
                    for p in possible_paths:
                        if os.path.exists(p):
                            logo_dir = p
                            break
                    
                    if logo_dir:
                        # 0. MAPA MANUAL
                        manual_map = {
                            "los ratones": "rat.png",
                            "karmine corp blue": "kcb.png",
                            "shifters": "shf.png",
                            "team heretics": "th.png",
                            "team vitality": "vit.png",
                            "sk gaming": "sk.png",
                            "fnatic": "fnc.png",
                            "g2 esports": "g2.png",
                            "giantx": "gx.png",
                            "karmine corp": "kc.png",
                            "mad lions koi": "mdk.png",
                            "movistar koi": "koi.png",  # AÑADIDO
                            "rogue": "rge.png",
                            "team bds": "bds.png",
                            "natus vincere": "navi.png"
                        }
                        
                        clean_name = name.lower().strip()
                        if clean_name in manual_map:
                             full_p = os.path.join(logo_dir, manual_map[clean_name])
                             if os.path.exists(full_p):
                                  final_static_url = f"/static/logos/{manual_map[clean_name]}"
                                  log(f"🎯 Logo MANUAL encontrado para {name}: {final_static_url}")
                        
                        # Solo buscamos automáticamente si no hemos encontrado el manual
                        if not final_static_url:
                            # 1. Probar codigo (fnc.png)
                            candidates = [
                                (code.lower() + ".png"),
                                (name.lower().replace(" ", "_").replace(".", "") + ".webp"),
                                (name.lower().replace(" ", "_") + ".webp"),
                                (name.lower().replace(" ", "_") + ".png")
                            ]
                            
                            # log(f"[DEBUG] Buscando logo local en {logo_dir} para {name} (Code: {code})")
                            for cand in candidates:
                                full_p = os.path.join(logo_dir, cand)
                                found = os.path.exists(full_p)
                                if found:
                                    final_static_url = f"/static/logos/{cand}"
                                    log(f"🎯 Logo local encontrado para {name}: {final_static_url}")
                                    break
                    else:
                         # log("[DEBUG] No se encontró directorio de logos en ninguna de las rutas esperadas.")
                         pass

                    img_url = ""
                    if final_static_url:
                        img_url = final_static_url
                    else:
                        # Si no hay local, intentamos scraping
                        try:
                            imgs = row.find_elements(By.TAG_NAME, "img")
                            if imgs: 
                                # Priorizar data-src si existe (lazy load), sino src
                                img_src = imgs[0].get_attribute("data-src") or imgs[0].get_attribute("src")
                                if img_src:
                                    img_url = img_src
                                    if not name: # Último intento de nombre
                                        name = imgs[0].get_attribute("alt")
                        except: pass
                        
                        # Fallback FINAL: Si no hay imagen, intentar construirla
                        # Gol.gg suele usar: https://gol.gg/_img/teams/Nombre.png
                        if not img_url and name:
                             img_url = f"https://gol.gg/_img/teams/{name}.png"
                             # log(f"[DEBUG] Usando fallback web para {name}: {img_url}")
                    
                    if not img_url: img_url = "" # Asegurar string vacía, nunca None/NaN
                    
                    data_list.append({
                        "id": team_id,
                        "name": name,
                        "code": code,
                        "image_url": img_url, # Será str vacía si falla, no NaN
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
