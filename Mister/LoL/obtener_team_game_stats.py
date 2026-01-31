import time
import re
import requests
import json
from bs4 import BeautifulSoup
from utils import log
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
            completed_ids = get_completed_matches(schema=schema, entity_type="MATCH_TEAM")
            matches_to_process = [m for m in all_matches if m.id not in completed_ids]
            log(f"Matches totales: {len(all_matches)}. Pendientes: {len(matches_to_process)}")

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

                blue_header = soup.find(class_="blue-line-header")
                if not blue_header:
                    log("    Headers no encontrados, skipping.")
                    continue
                
                blue_head_text = blue_header.get_text(strip=True)
                blue_name = blue_head_text.split("-")[0].strip()
                blue_win = "WIN" in blue_head_text

                red_header = soup.find(class_="red-line-header")
                red_head_text = red_header.get_text(strip=True)
                red_name = red_head_text.split("-")[0].strip()
                red_win = "WIN" in red_head_text

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
                        g_str = boxes[4].get_text().lower().replace('k', '').strip()
                        if '.' in g_str: g = int(float(g_str) * 1000)
                        else: g = int(g_str) 
                    except: g = 0
                    return k, t, d, b, g, elder_count

                b_kills, b_towers, b_dragons, b_barons, b_gold, b_elder = get_basic_stats(blue_header, "blue_line")
                r_kills, r_towers, r_dragons, r_barons, r_gold, r_elder = get_basic_stats(red_header, "red_line")

                # Timeline Logic (Simplified for brevity, similar structure to original)
                b_fb, b_ft, r_fb, r_ft = False, False, False, False
                b_fb_time, b_ft_time, b_fd_time, b_fb_time_baron = None, None, None, None
                r_fb_time, r_ft_time, r_fd_time, r_fb_time_baron = None, None, None, None
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
                                                if b_fb_time_baron is None: b_fb_time_baron = sec
                                            else:
                                                if r_fb_time_baron is None: r_fb_time_baron = sec
                                        elif "dragon" in alt or "drake" in alt:
                                            dtype = "Unknown"
                                            for k in ["elder","hextech","chemtech","cloud","mountain","ocean","infernal","fire"]:
                                                if k in alt: dtype = k.capitalize()
                                            evt = {"type": dtype, "time": sec}
                                            if is_blue:
                                                b_dragon_events.append(evt)
                                                if b_fd_time is None: b_fd_time = sec
                                            else:
                                                r_dragon_events.append(evt)
                                                if r_fd_time is None: r_fd_time = sec

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

                row_blue = create_team_row(match.blue_team_id, "Blue", blue_win, b_kills, b_gold, b_towers, b_dragons, b_barons, b_elder, b_fb, b_ft, b_fb_time, b_ft_time, b_fd_time, b_fb_time_baron, b_dragon_events, b_grubs, b_plates, b_p_top, b_p_mid, b_p_bot, blue_vis, blue_jg, b_bans, b_picks)
                row_red = create_team_row(match.red_team_id, "Red", red_win, r_kills, r_gold, r_towers, r_dragons, r_barons, r_elder, r_fb, r_ft, r_fb_time, r_ft_time, r_fd_time, r_fb_time_baron, r_dragon_events, r_grubs, r_plates, r_p_top, r_p_mid, r_p_bot, red_vis, red_jg, r_bans, r_picks)
                
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
