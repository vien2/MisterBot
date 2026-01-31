import datetime
import urllib.parse
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils import log

def obtener_tournaments(driver, schema=None, target_season="S16", **kwargs):
    """
    Scrapes tournaments from gol.gg using the provided driver.
    Returns a LIST OF DICTIONARIES for ETL processing.
    """
    log(f"Iniciando Scraper de Torneos para Temporada {target_season}...")
    data_list = []

    try:
        url = "https://gol.gg/tournament/list/"
        log(f"Navegando a {url}...")
        driver.get(url)

        wait = WebDriverWait(driver, 30)
        table = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "table_list")))
        
        rows = table.find_elements(By.TAG_NAME, "tr")
        log(f"Filas encontradas: {len(rows)}")
        
        # Skip header
        for row in rows[1:]:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) < 6:
                continue
            
            # Extract Name & Link
            name_col = cols[1]
            try:
                link_el = name_col.find_element(By.TAG_NAME, "a")
                name = link_el.text.strip()
                href = link_el.get_attribute("href")
            except:
                continue
                
            # Extract Slug
            slug_raw = href.rstrip("/").split("/")[-1]
            slug = urllib.parse.unquote(slug_raw)
            
            # Extract Region
            region = cols[2].text.strip()
            
            # Dates
            start_str = cols[5].text.strip()
            end_str = cols[6].text.strip()
            
            # Standard formatting for CSV/DB
            # DB expects YYYY-MM-DD usually
            
            row_dict = {
                "id": slug,
                "name": name,
                "slug": slug,
                "region": region,
                "season": target_season,
                "start_date": start_str if start_str != "-" else None,
                "end_date": end_str if end_str != "-" else None
            }
            data_list.append(row_dict)
            
        log(f"Scraping Completado. Torneos extraídos: {len(data_list)}")
        return data_list

    except Exception as e:
        log(f"Error en Scraper de Torneos: {e}")
        return []
