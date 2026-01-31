import time
import re
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils import log
from .utils_lol import get_match_ids, get_completed_matches, obtener_o_crear_jugador, log_scrape_status

def obtener_game_stats(driver, schema="LoL_Stats", **kwargs):
    """
    Scrapes full game stats using Selenium.
    Returns List[dict] for ETL. NO ORM.
    """
    log("Iniciando Scraper de Game Stats (Full Stats - No ORM)...")
    data_list = []

    try:
        target_ids = kwargs.get('target_ids', [])
        
        matches = []
        if target_ids:
            matches = get_match_ids(target_ids, schema=schema)
        else:
            all_matches = get_match_ids(schema=schema)
            # Check completed matches in log
            completed_ids = get_completed_matches(schema=schema, entity_type="MATCH")
            matches = [m for m in all_matches if m.id not in completed_ids]
            log(f"Matches totales: {len(all_matches)}. Pendientes: {len(matches)}")

        if not matches:
            log("No hay matches pendientes.")
            return []

        total_processed = 0

        for i, match in enumerate(matches):
            log(f"[{i+1}/{len(matches)}] Match {match.id}")
            
            try:
                url = f"https://gol.gg/game/stats/{match.id}/page-fullstats/"
                driver.get(url)
                
                wait = WebDriverWait(driver, 10)
                wait.until(EC.presence_of_element_located((By.CLASS_NAME, "completestats")))
                time.sleep(1) 
                
                # Setup side lookup
                # blue_team_id_for_scrape = match.blue_team_id ... (Actually we pass IDs to player creation)
                
                table = driver.find_element(By.CLASS_NAME, "completestats")
                tbody = table.find_element(By.TAG_NAME, "tbody")
                rows = tbody.find_elements(By.TAG_NAME, "tr")
                thead = table.find_element(By.TAG_NAME, "thead")
                headers = thead.find_elements(By.TAG_NAME, "th")
                
                col_side_map = {}
                col_role_map = {}
                player_map = {} # {col_idx: player_id}
                champ_map = {}
                
                # Headers
                for idx, th in enumerate(headers):
                    if idx == 0: continue
                    try:
                        img = th.find_element(By.TAG_NAME, "img")
                        champ_map[idx] = img.get_attribute("alt")
                    except:
                        champ_map[idx] = "Unknown"

                # Parse Table logic
                data_rows = {}
                
                for row in rows:
                    cols = row.find_elements(By.TAG_NAME, "td")
                    if not cols: continue
                    metric = cols[0].get_attribute("textContent").strip()
                    
                    if metric == "Role":
                        for c_idx, c in enumerate(cols[1:], start=1):
                            cls, txt = c.get_attribute("class"), c.get_attribute("textContent").strip()
                            col_role_map[c_idx] = txt
                            if "blue" in cls: col_side_map[c_idx] = "Blue"
                            elif "red" in cls: col_side_map[c_idx] = "Red"
                            else: col_side_map[c_idx] = "Blue" if c_idx <= 5 else "Red"
                    
                    elif metric == "Player":
                         for c_idx, c in enumerate(cols[1:], start=1):
                            pid = 0
                            p_name = "Unknown"
                            # Determine intended Team ID based on column side
                            c_side = col_side_map.get(c_idx, "Blue" if c_idx <= 5 else "Red")
                            team_id = match.blue_team_id if c_side == "Blue" else match.red_team_id
                             
                            try:
                                lnk = c.find_elements(By.TAG_NAME, "a")
                                if lnk:
                                    p_name = lnk[0].get_attribute("textContent").strip()
                                    # Try extracting PID from href? Or just lookup by name.
                                    # Using name lookup is safer with our helper.
                                else:
                                    p_name = c.get_attribute("textContent").strip()

                                # Resolve Role
                                role_val = col_role_map.get(c_idx, "Unknown")
                                if role_val == "Unknown":
                                    role_indices = ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"]
                                    role_val = role_indices[(c_idx - 1) % 5]
                                
                                # GET OR CREATE PLAYER (Raw SQL)
                                pid = obtener_o_crear_jugador(p_name, team_id, role_val, schema=schema)
                                
                            except Exception as e:
                                log(f"    Error parsing player col {c_idx}: {e}")
                            
                            player_map[c_idx] = pid
                    else:
                        vals = []
                        for c in cols[1:]: vals.append(c.get_attribute("textContent").strip())
                        data_rows[metric] = vals

                # Construct Dicts
                for c_idx in range(1, 11):
                    side = col_side_map.get(c_idx, "Blue" if c_idx <= 5 else "Red")
                    pid = player_map.get(c_idx, 0)
                    if not pid: continue
                    
                    champ_raw = champ_map.get(c_idx, "Unknown")
                    champ = champ_raw.replace("'", "").replace(".", "").replace(" ", "")
                    if "&" in champ: champ = champ.split("&")[0]

                    def get_val(key, type_func=int, default=0):
                        if key not in data_rows: return default
                        raw = data_rows[key][c_idx-1]
                        try:
                            if "%" in raw: return float(raw.replace("%", "").strip()) / 100.0
                            if "k" in raw and type_func == int: return int(float(raw.replace("k", "")) * 1000)
                            if type_func == int: return int(float(raw))
                            return type_func(raw)
                        except: return default
                    
                    is_p_win = (match.winner_id == (match.blue_team_id if side == "Blue" else match.red_team_id)) if match.winner_id else False

                    row_dict = {
                        "match_id": match.id,
                        "player_id": pid,
                        "champion_name": champ,
                        "side": side,
                        "win": is_p_win, # True/False handled correctly by psycopg2? Yes usually maps to boolean.
                        
                        "kills": get_val("Kills"),
                        "deaths": get_val("Deaths"),
                        "assists": get_val("Assists"),
                        "total_gold": get_val("Golds"),
                        "cs": get_val("CS"),
                        "level": get_val("Level"),
                        
                        "vision_score": get_val("Vision Score"),
                        "wards_placed": get_val("Wards placed"),
                        "wards_killed": get_val("Wards destroyed"),
                        "control_wards_purchased": get_val("Control Wards Purchased"),
                        "detector_wards_placed": get_val("Detector Wards Placed"),
                        
                        "damage_dealt": get_val("Total damage to Champion"),
                        "physical_damage_dealt_to_champions": get_val("Physical Damage"),
                        "magic_damage_dealt_to_champions": get_val("Magic Damage"),
                        "true_damage_dealt_to_champions": get_val("True Damage"),
                        "damage_taken": get_val("Total damage taken"),
                        "damage_self_mitigated": get_val("Damage self mitigated"),
                        "damage_dealt_to_turrets": get_val("Damage dealt to turrets"),
                        
                        "gold_per_min": float(get_val("GPM", float)),
                        "gold_share": get_val("GOLD%", float),
                        "cs_per_min": get_val("CSM", float),

                        "gold_diff_15": get_val("GD@15"),
                        "cs_diff_15": get_val("CSD@15"),
                        "xp_diff_15": get_val("XPD@15"),
                        "level_diff_15": get_val("LVLD@15"),

                        "kill_participation": get_val("KP%", float),
                        "solo_kills": get_val("Solo kills"),
                        
                        "time_ccing_others": get_val("Time ccing others"),
                        "total_time_cc_dealt": get_val("Total Time CC Dealt"),
                        "total_heal": get_val("Total heal"),
                        "total_heals_on_teammates": get_val("Total Heals On Teammates"),
                        "damage_share": get_val("DMG%", float),
                        "vision_share": get_val("VS%", float),
                        "vision_score_per_minute": get_val("VSPM", float),
                        "wards_per_minute": get_val("WPM", float),
                        "wards_cleared_per_minute": get_val("WCPM", float),
                        
                        "cs_in_team_jungle": get_val("CS in Team's Jungle"),
                        "cs_in_enemy_jungle": get_val("CS in Enemy Jungle")
                    }
                    data_list.append(row_dict)
                
                log_scrape_status(match.id, "MATCH", "COMPLETED", "Scraped OK")
                total_processed += 1
                time.sleep(1)

            except Exception as e:
                log(f"    Error scraping match {match.id}: {e}")
                log_scrape_status(match.id, "MATCH", "FAILED", str(e)[:300])

        return data_list

    except Exception as overall_ex:
        log(f"Critical Scraper Error: {overall_ex}")
        return []
