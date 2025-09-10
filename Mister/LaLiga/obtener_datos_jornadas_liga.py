
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from utils import log
from bs4 import BeautifulSoup

def obtener_datos_jornadas_liga(driver, schema=None):
    _ = schema
    log("obtener_datos_jornadas_liga: Inicio de la función")

    datos_jornadas = []
    wait = WebDriverWait(driver, 10)

    try:
        # 1️⃣ Ir a la sección de competición (igual que en obtener_datos_liga)
        boton_jornada = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "div.feed-top-gameweek button.btn.btn-sw")
        ))
        boton_jornada.click()
        log("obtener_datos_jornadas_liga: Botón de jornada clickeado")
        time.sleep(2)

        boton_competicion = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "div.sw-top-right button[data-sw='competition']")
        ))
        boton_competicion.click()
        log("obtener_datos_jornadas_liga: Botón 'Competición' clickeado")
        time.sleep(2)

        # 2️⃣ Obtener el HTML y parsear jornadas
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, "html.parser")

        jornadas_contenedores = soup.find_all("div", class_="box box-matches")
        log(f"obtener_datos_jornadas_liga: {len(jornadas_contenedores)} jornadas encontradas")

        for contenedor in jornadas_contenedores:
            # Extraer número de jornada desde el h3
            titulo = contenedor.find("h3")
            if not titulo:
                continue
            jornada_texto = titulo.get_text(strip=True)  # Ej: "Jornada 1"
            jornada_numero = int(jornada_texto.replace("Jornada", "").strip())

            partidos = contenedor.find_all("li")
            log(f"obtener_datos_jornadas_liga: Jornada {jornada_numero} - {len(partidos)} partidos encontrados")

            for partido in partidos:
                try:
                    home_div = partido.find("div", class_="home")
                    away_div = partido.find("div", class_="away")
                    mid_div = partido.find("div", class_="mid score")

                    equipo_local = home_div.get_text(" ", strip=True).replace("\n", " ")
                    equipo_visitante = away_div.get_text(" ", strip=True).replace("\n", " ")
                    resultado = mid_div.get_text(strip=True) if mid_div else "N/A"

                    registro = {
                        "jornada": jornada_numero,
                        "local": equipo_local,
                        "visitante": equipo_visitante,
                        "resultado": resultado,
                    }
                    datos_jornadas.append(registro)

                except Exception as e:
                    log(f"obtener_datos_jornadas_liga: Error procesando partido en Jornada {jornada_numero}: {e}")
                    continue

        log(f"obtener_datos_jornadas_liga: Finalización con {len(datos_jornadas)} partidos procesados")
        return datos_jornadas

    except TimeoutException:
        log("obtener_datos_jornadas_liga: Timeout al cargar los elementos de la página")
        return []
    except NoSuchElementException:
        log("obtener_datos_jornadas_liga: No se encontró un elemento esperado")
        return []
    except Exception as e:
        log(f"obtener_datos_jornadas_liga: Error inesperado: {e}")
        return []
def obtener_datos_jornadas_liga(driver, schema=None):
    _ = schema
    log("obtener_datos_jornadas_liga: Inicio de la función")

    datos_jornadas = []
    wait = WebDriverWait(driver, 10)

    try:
        # 1️⃣ Ir a la sección de competición (igual que en obtener_datos_liga)
        boton_jornada = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "div.feed-top-gameweek button.btn.btn-sw")
        ))
        boton_jornada.click()
        log("obtener_datos_jornadas_liga: Botón de jornada clickeado")
        time.sleep(2)

        boton_competicion = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "div.sw-top-right button[data-sw='competition']")
        ))
        boton_competicion.click()
        log("obtener_datos_jornadas_liga: Botón 'Competición' clickeado")
        time.sleep(2)

        # 2️⃣ Obtener el HTML y parsear jornadas
        html_content = driver.page_source
        soup = BeautifulSoup(html_content, "html.parser")

        jornadas_contenedores = soup.find_all("div", class_="box box-matches")
        log(f"obtener_datos_jornadas_liga: {len(jornadas_contenedores)} jornadas encontradas")

        for contenedor in jornadas_contenedores:
            # Extraer número de jornada desde el h3
            titulo = contenedor.find("h3")
            if not titulo:
                continue
            jornada_texto = titulo.get_text(strip=True)  # Ej: "Jornada 1"
            jornada_numero = int(jornada_texto.replace("Jornada", "").strip())

            partidos = contenedor.find_all("li")
            log(f"obtener_datos_jornadas_liga: Jornada {jornada_numero} - {len(partidos)} partidos encontrados")

            for partido in partidos:
                try:
                    home_div = partido.find("div", class_="home")
                    away_div = partido.find("div", class_="away")
                    mid_div = partido.find("div", class_="mid score")

                    equipo_local = home_div.get_text(" ", strip=True).replace("\n", " ")
                    equipo_visitante = away_div.get_text(" ", strip=True).replace("\n", " ")
                    resultado = mid_div.get_text(strip=True) if mid_div else "N/A"

                    registro = {
                        "jornada": jornada_numero,
                        "local": equipo_local,
                        "visitante": equipo_visitante,
                        "resultado": resultado,
                    }
                    datos_jornadas.append(registro)

                except Exception as e:
                    log(f"obtener_datos_jornadas_liga: Error procesando partido en Jornada {jornada_numero}: {e}")
                    continue

        log(f"obtener_datos_jornadas_liga: Finalización con {len(datos_jornadas)} partidos procesados")
        return datos_jornadas

    except TimeoutException:
        log("obtener_datos_jornadas_liga: Timeout al cargar los elementos de la página")
        return []
    except NoSuchElementException:
        log("obtener_datos_jornadas_liga: No se encontró un elemento esperado")
        return []
    except Exception as e:
        log(f"obtener_datos_jornadas_liga: Error inesperado: {e}")
        return []
