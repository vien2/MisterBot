import time
import re
import requests
import json
from bs4 import BeautifulSoup
from utils import log, conexion_db
from .utils_lol import get_match_ids, get_completed_matches, log_scrape_status

def obtener_team_game_stats(driver, schema="LoL_Stats", **kwargs):
    """
    Scrapes team stats via Requests. 
    Returns List[dict] for CSV/DB loading.
    """
    log("Iniciando Scraper de Team Game Stats (Requests - No ORM)...")
    data_list = []

    try:
        target_ids = kwargs.get('target_ids', [])
        
        matches_to_process = []
        if target_ids:
            matches_to_process = get_match_ids(target_ids, schema=schema)
        else:
            all_matches = get_match_ids(schema=schema)
            # Check completed matches using DIRECT DB CHECK (Smart Check)
            # We want patches that DO NOT have records in lol_stats.team_game_stats
            # But we also want to re-scrape if stats are "empty" (gold=0), so we only count VALID stats
            with conexion_db() as conn:
                with conn.cursor() as cur:
                    # Only consider it "done" if we have stats for BOTH teams (count >= 2) and gold > 0
                    cur.execute(f"""
                        SELECT match_id 
                        FROM {schema}.team_game_stats 
                        WHERE total_gold > 1000 
                        GROUP BY match_id 
                        HAVING count(*) >= 2
                    """)
                    existing_ids = {r[0] for r in cur.fetchall()}
            
            matches_to_process = [m for m in all_matches if m.id not in existing_ids]
            log(f"Matches totales: {len(all_matches)}. Con Stats Validos: {len(existing_ids)}. Pendientes reales: {len(matches_to_process)}")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        total_processed = 0
        
        for i, match in enumerate(matches_to_process):
            log(f"[{i+1}/{len(matches_to_process)}] Scraping Match {match.id}")
            
            try:
                url = f"https://gol.gg/game/stats/{match.id}/page-game/"
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code != 200:
                    log(f"    Saltando {match.id}: Status {r.status_code}")
                    continue
                    
                html_source = r.text
                soup = BeautifulSoup(html_source, 'html.parser')

                # --- 1. GAME DURATION & PATCH ---
                duration_sec = 0
                time_div = soup.find(string="Game Time")
                if time_div:
                    parent_div = time_div.find_parent('div')
                    if parent_div:
                        h1 = parent_div.find('h1')
                        if h1:
                            dur_text = h1.get_text(strip=True)
                            parts = dur_text.split(':')
                            if len(parts) == 2:
                                try:
                                    duration_sec = int(parts[0]) * 60 + int(parts[1])
                                except: pass

                # --- VALIDATE MATCH STATUS ---
                # Should not process matches with 0 duration or unplayed
                if duration_sec < 300: # Less than 5 mins implies remake or unplayed
                     log(f"    Skipping {match.id}: Invalid Duration ({duration_sec}s). Likely unplayed.")
                     continue

                patch_text = "Unknown"
                patch_div = soup.find(lambda tag: tag.name == "div" and "col-3" in tag.get("class", []) and " v" in tag.get_text())
                if patch_div:
                    patch_text = patch_div.get_text(strip=True).replace('v', '')

                # --- 2. EXTRACT JS DATA ---
                def get_js_dataset(var_name, source, label_keyword):
                    block_match = re.search(r"var\s+" + var_name + r"\s*=\s*(\{[\s\S]*?\});", source)
                    if not block_match: return None
                    block = block_match.group(1)
                    safe_key = re.escape(label_keyword)
                    p = r"label:\s*['\"][^'\"]*" + safe_key + r"[^'\"]*['\"][\s\S]*?data\s*:\s*\[([\d\.,\s]+)\]"
                    m = re.search(p, block)
                    if m:
                         vals_str = m.group(1)
                         try: return [float(x.strip()) for x in vals_str.split(',') if x.strip()]
                         except: return [0,0]
                    return None

                # --- 2. EXTRACT TEAM DATA AND WINNER ---
                # We find the headers based on text "Blue side" and "Red side" or by finding the DIVs with specific classes
                # But better: Gol.gg usually has two main headers. One for Blue (left), one for Red (right).
                
                def parse_header(header_div):
                    if not header_div: return None, None, False
                    link = header_div.find("a")
                    name = ""
                    t_id = 0
                    if link:
                        name = link.get_text(strip=True)
                        href = link.get("href")
                        try: t_id = int(href.split("stats/")[1].split("/")[0])
                        except: pass
                    else:
                        name = header_div.get_text(strip=True).split("-")[0].strip()
                    
                    is_win = "WIN" in header_div.get_text(strip=True)
                    return name, t_id, is_win

                blue_header = soup.find(class_="blue-line-header")
                red_header = soup.find(class_="red-line-header")
                
                blue_name, blue_t_id, blue_win = parse_header(blue_header)
                red_name, red_t_id, red_win = parse_header(red_header)
                
                if not blue_t_id: 
                    blue_t_id = match.blue_team_id
                    # Extra Check: If we found Red but not Blue, and match has IDs, ensure we don't swap them
                    if not blue_t_id and red_t_id and match.blue_team_id == red_t_id:
                         blue_t_id = match.red_team_id # Swap detected? logic is complex here.
                         # Better: Just trust Match object if scraper failed
                
                if not red_t_id: 
                    red_t_id = match.red_team_id

                # If still 0, we can't save correctly, but better to save with 0 than skip

                # --- 2.1 UPDATE PARENT MATCH RECORD ---
                # This is CRITICAL: If the match was discovered as "empty", we must fill it now
                # so that obtain_game_stats and the Web App know which teams played.
                try:
                    with conexion_db() as conn:
                        with conn.cursor() as cur:
                            winner_id = blue_t_id if blue_win else (red_t_id if red_win else None)
                            cur.execute(f"""
                                UPDATE {schema}.matches 
                                SET blue_team_id = %s, red_team_id = %s, winner_id = %s 
                                WHERE id = %s
                            """, (blue_t_id, red_t_id, winner_id, match.id))
                        conn.commit()
                except Exception as e_upd:
                    log(f"    Warning: No se pudo actualizar el match {match.id}: {e_upd}")

                blue_vis = get_js_dataset("visionData", html_source, blue_name) or [0,0]
                red_vis = get_js_dataset("visionData", html_source, red_name) or [0,0]
                
                def get_counter_data(source):
                     block_match = re.search(r"var\s+counterData\s*=\s*(\{[\s\S]*?\});", source)
                     if not block_match: return ([0,0], [0,0])
                     block = block_match.group(1)
                     matches_arr = re.findall(r"data\s*:\s*\[([\d\.,\s]+)\]", block)
                     if len(matches_arr) >= 2:
                         b = [float(x) for x in matches_arr[0].split(',')]
                         r = [float(x) for x in matches_arr[1].split(',')]
                         return (b, r)
                     return ([0,0], [0,0])

                blue_jg, red_jg = get_counter_data(html_source)

                # --- 3. BANS & PICKS ---
                def get_champs_list(header_elem, section_name):
                    if not header_elem: return ""
                    container = header_elem.find_parent("div", class_="col-sm-6")
                    if not container: return ""
                    label = container.find(string=lambda t: t and section_name in t)
                    if not label: return ""
                    label_parent = label.find_parent('div')
                    if not label_parent: return ""
                    champlist_div = label_parent.find_next_sibling('div')
                    if not champlist_div: return ""
                    imgs = champlist_div.find_all('img')
                    names = []
                    for img in imgs:
                        t = img.get('title', '').replace(" stats", "").strip() or img.get('alt', '').replace(" stats", "").strip()
                        if t: names.append(t)
                    return ",".join(names)
                
                b_bans = get_champs_list(blue_header, "Bans")
                b_picks = get_champs_list(blue_header, "Picks")
                r_bans = get_champs_list(red_header, "Bans")
                r_picks = get_champs_list(red_header, "Picks")

                # --- 4. PLATES & VOIDGRUBS ---
                b_grubs, r_grubs = 0, 0
                b_plates, r_plates = 0, 0
                b_p_top, r_p_top = 0, 0
                b_p_mid, r_p_mid = 0, 0
                b_p_bot, r_p_bot = 0, 0
                
                void_elem = soup.find(string=re.compile("Voidgrubs"))
                if void_elem:
                    void_col = void_elem.find_parent("div", class_=re.compile("col-"))
                    if void_col:
                         void_row = void_col.find_parent("div", class_="row")
                         if void_row:
                             main_col = void_row.find_parent("div", class_="col-12")
                             if main_col:
                                for r_row in main_col.find_all("div", class_="row"):
                                    cols = r_row.find_all("div", recursive=False)
                                    if len(cols) < 3: continue
                                    label = cols[0].get_text(strip=True)
                                    try: v1 = int(cols[1].get_text(strip=True))
                                    except: v1=0
                                    try: v2 = int(cols[2].get_text(strip=True))
                                    except: v2=0
                                    
                                    if "Voidgrubs" in label: b_grubs, r_grubs = v1, v2
                                    elif "Plates" in label and "TOP" not in label: b_plates, r_plates = v1, v2
                                    elif "Plates TOP" in label: b_p_top, r_p_top = v1, v2
                                    elif "Plates MID" in label: b_p_mid, r_p_mid = v1, v2
                                    elif "Plates BOT" in label: b_p_bot, r_p_bot = v1, v2

                def get_basic_stats(header_elem, color_class):
                    if not header_elem: return (0,0,0,0,0,0)
                    container = header_elem.find_parent("div", class_="col-sm-6")
                    if not container: return (0,0,0,0,0,0)
                    boxes = container.find_all(class_=f"score-box {color_class}")
                    if len(boxes) < 5: return (0,0,0,0,0,0)
                    
                    def parse_box(idx):
                        try: return int(re.search(r'\d+', boxes[idx].get_text()).group())
                        except: return 0
                    
                    k, t, d, b = parse_box(0), parse_box(1), parse_box(2), parse_box(3)
                    
                    elder_count = 0
                    db_parent = boxes[2].find_parent('div')
                    if db_parent:
                        elder_count = len(db_parent.find_all("img", alt=re.compile("Elder", re.IGNORECASE)))

                    try:
                        g_str = boxes[4].get_text().lower().strip()
                        if 'k' in g_str:
                             g_val = float(g_str.replace('k', ''))
                             g = int(g_val * 1000)
                        else:
                             g = int(float(g_str)) # fallback if no k
                    except: g = 0
                    return k, t, d, b, g, elder_count

                b_kills, b_towers, b_dragons, b_barons, b_gold, b_elder = get_basic_stats(blue_header, "blue_line")
                r_kills, r_towers, r_dragons, r_barons, r_gold, r_elder = get_basic_stats(red_header, "red_line")

                # Timeline Logic
                b_fb, b_ft, r_fb, r_ft = False, False, False, False
                # FIX: Init to 0 instead of None to avoid Pandas 'nan' float in Integer columns
                b_fb_time, b_ft_time, b_fd_time, b_fb_time_baron = 0, 0, 0, 0
                r_fb_time, r_ft_time, r_fd_time, r_fb_time_baron = 0, 0, 0, 0
                b_dragon_events, r_dragon_events = [], []

                timeline_container = soup.find("div", class_="flex-wrap")
                if timeline_container:
                    def parse_time(t_str):
                        try:
                            parts = t_str.strip().split(':')
                            return int(parts[0]) * 60 + int(parts[1])
                        except: return 0

                    for action in timeline_container.find_all("span", style="display:inline-block"):
                         for cls, is_blue in [("blue_action", True), ("red_action", False)]:
                            act = action.find("span", class_=cls)
                            if act:
                                txt = act.get_text(separator="|").strip()
                                m = re.search(r"(\d+:\d+)", txt)
                                if m:
                                    sec = parse_time(m.group(1))
                                    img = act.find("img")
                                    if img:
                                        alt = img.get("alt", "").lower()
                                        if "first blood" in alt:
                                            if is_blue: b_fb, b_fb_time = True, sec
                                            else: r_fb, r_fb_time = True, sec
                                        elif "first tower" in alt:
                                            if is_blue: b_ft, b_ft_time = True, sec
                                            else: r_ft, r_ft_time = True, sec
                                        elif "nashor" in alt or "baron" in alt:
                                            if is_blue: 
                                                if b_fb_time_baron == 0: b_fb_time_baron = sec
                                            else:
                                                if r_fb_time_baron == 0: r_fb_time_baron = sec
                                        elif "dragon" in alt or "drake" in alt:
                                            dtype = "Unknown"
                                            for k in ["elder","hextech","chemtech","cloud","mountain","ocean","infernal","fire"]:
                                                if k in alt: dtype = k.capitalize()
                                            evt = {"type": dtype, "time": sec}
                                            if is_blue:
                                                b_dragon_events.append(evt)
                                                if b_fd_time == 0: b_fd_time = sec
                                            else:
                                                r_dragon_events.append(evt)
                                                if r_fd_time == 0: r_fd_time = sec

                # Construct Dictionaries
                def create_team_row(team_id, side, win, k, g, t, d, b, e, fb, ft, fbt, ftt, fdt, fbtb, devs, grubs, plates, ptop, pmid, pbot, vis, jg, bans, picks):
                     return {
                        "match_id": match.id,
                        "team_id": team_id,
                        "side": side,
                        "win": win,
                        "game_duration": duration_sec,
                        "patch": patch_text,
                        "bans": bans,
                        "picks": picks,
                        "total_kills": k,
                        "total_gold": g,
                        "towers_destroyed": t,
                        "dragons_killed": d,
                        "barons_killed": b,
                        "elder_dragons_killed": e,
                        "first_blood": fb,
                        "first_tower": ft,
                        "first_blood_time": fbt,
                        "first_tower_time": ftt,
                        "first_dragon_time": fdt,
                        "first_baron_time": fbtb,
                        "dragon_events": json.dumps(devs),
                        "void_grubs": grubs,
                        "rift_heralds": 0,
                        "plates_total": plates,
                        "plates_top": ptop,
                        "plates_mid": pmid,
                        "plates_bot": pbot,
                        "wards_destroyed": int(vis[0]),
                        "wards_placed": int(vis[1]),
                        "jungle_share_15": jg[0],
                        "jungle_share_end": jg[1]
                     }

                row_blue = create_team_row(blue_t_id, "Blue", blue_win, b_kills, b_gold, b_towers, b_dragons, b_barons, b_elder, b_fb, b_ft, b_fb_time, b_ft_time, b_fd_time, b_fb_time_baron, b_dragon_events, b_grubs, b_plates, b_p_top, b_p_mid, b_p_bot, blue_vis, blue_jg, b_bans, b_picks)
                row_red = create_team_row(red_t_id, "Red", red_win, r_kills, r_gold, r_towers, r_dragons, r_barons, r_elder, r_fb, r_ft, r_fb_time, r_ft_time, r_fd_time, r_fb_time_baron, r_dragon_events, r_grubs, r_plates, r_p_top, r_p_mid, r_p_bot, red_vis, red_jg, r_bans, r_picks)
                
                # Add Deaths/Assists cross-ref (optional, simple logic here assumes mirrored kills)
                row_blue["total_deaths"], row_blue["total_assists"] = r_kills, 0
                row_red["total_deaths"], row_red["total_assists"] = b_kills, 0

                data_list.append(row_blue)
                data_list.append(row_red)
                
                log_scrape_status(match.id, "MATCH_TEAM", "COMPLETED", "Scraped OK")
                total_processed += 1
                time.sleep(0.1)

            except Exception as e:
                log(f"    Error scraping match {match.id}: {e}")
                log_scrape_status(match.id, "MATCH_TEAM", "FAILED", str(e)[:300])

        return data_list

    except Exception as overall_ex:
        log(f"Critical Scraper Error: {overall_ex}")
        return []
