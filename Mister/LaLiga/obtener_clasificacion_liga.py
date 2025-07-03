from bs4 import BeautifulSoup
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from utils import log

def obtener_datos_liga(driver,schema=None):
    _ = schema
    log("obtener_datos_liga: Inicio de la función")

    wait = WebDriverWait(driver, 10)

    try:
        enlace_mas = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Más')]/parent::li/a")))
        enlace_mas.click()
        log("obtener_datos_liga: Enlace 'Más' clickeado")
        time.sleep(2)

        enlace_laliga = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'LaLiga')]")))
        enlace_laliga.click()
        log("obtener_datos_liga: Botón 'LaLiga' clickeado")
        time.sleep(2)

        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')

        table = soup.find('div', class_='box box-table')
        if not table:
            log("obtener_datos_liga: No se encontró la tabla de clasificación")
            return []

        rows = table.find_all('tr')[1:]

        data_list = []

        for row in rows:
            cols = row.find_all('td')
            try:
                equipo_data = {
                    "Posicion": cols[0].text.strip(),
                    "Escudo": cols[1].find('img')['src'],
                    "Equipo": cols[2].text.strip(),
                    "PTS": cols[3].text.strip(),
                    "PJ": cols[4].text.strip(),
                    "PG": cols[5].text.strip(),
                    "PE": cols[6].text.strip(),
                    "PP": cols[7].text.strip(),
                    "DG": cols[8].text.strip() if len(cols) > 8 else 'N/A'
                }
                data_list.append(equipo_data)
                #log(f"obtener_datos_liga: Datos añadidos - {equipo_data['Equipo']} | PTS: {equipo_data['PTS']}")
            except Exception as e:
                log(f"obtener_datos_liga: Error procesando una fila de la tabla: {e}")
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