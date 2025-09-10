from bs4 import BeautifulSoup
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from utils import log

def obtener_datos_liga(driver, schema=None):
    _ = schema
    log("obtener_datos_liga: Inicio de la función")

    wait = WebDriverWait(driver, 10)

    try:
        # 1️⃣ Clic en el botón de la jornada
        boton_jornada = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "div.feed-top-gameweek button.btn.btn-sw")
        ))
        boton_jornada.click()
        log("obtener_datos_liga: Botón de jornada clickeado")
        time.sleep(2)

        # 2️⃣ Clic en el botón de competición (calendario/tabla)
        boton_competicion = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "div.sw-top-right button[data-sw='competition']")
        ))
        boton_competicion.click()
        log("obtener_datos_liga: Botón 'Competición' clickeado")
        time.sleep(2)

        # 3️⃣ Obtener HTML y buscar la tabla
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, "html.parser")

        table = soup.find("div", class_="box box-table")
        if not table:
            log("obtener_datos_liga: No se encontró la tabla de clasificación")
            return []

        rows = table.find_all("tr")[1:]  # Omitimos la cabecera

        data_list = []
        for row in rows:
            cols = row.find_all("td")
            try:
                equipo_data = {
                    "Posicion": cols[0].get_text(strip=True),
                    "Escudo": cols[1].find("img")["src"],
                    "Equipo": cols[2].get_text(strip=True),
                    "PTS": cols[3].get_text(strip=True),
                    "PJ": cols[4].get_text(strip=True),
                    "PG": cols[5].get_text(strip=True),
                    "PE": cols[6].get_text(strip=True),
                    "PP": cols[7].get_text(strip=True),
                    "DG": cols[8].get_text(strip=True) if len(cols) > 8 else "N/A",
                }
                data_list.append(equipo_data)
            except Exception as e:
                log(f"obtener_datos_liga: Error procesando fila: {e}")
                continue

        log(f"obtener_datos_liga: Finalización exitosa con {len(data_list)} equipos procesados")
        return data_list

    except TimeoutException:
        log("obtener_datos_liga: TimeoutException - No se pudo navegar correctamente")
        return None
    except NoSuchElementException:
        log("obtener_datos_liga: NoSuchElementException - Elemento no encontrado durante la navegación")
        return None
    except Exception as e:
        log(f"obtener_datos_liga: Error inesperado: {e}")
        return None
