from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils import log

def obtener_mercado(driver, schema=None):
    _ = schema
    log("obtener_mercado: Inicio de la función")

    datos_mercado = []
    wait = WebDriverWait(driver, 10)

    # 1. Clic en pestaña Mercado
    try:
        enlace_mercado = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "li[data-pag='market'] a.navbar-switch-tab"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", enlace_mercado)
        enlace_mercado.click()
        log("obtener_mercado: Enlace 'Mercado' clickeado")
    except Exception as e:
        log(f"obtener_mercado: Error al hacer clic en 'Mercado': {e}")
        return datos_mercado

    # 2. Abrir filtro y seleccionar "Libres"
    try:
        filtrar = wait.until(
            EC.element_to_be_clickable((By.ID, "btn-filter-market"))
        )
        filtrar.click()
        log("obtener_mercado: Botón 'Filtrar' clickeado")

        libres = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//label[@for='seg-owner-0']"))
        )
        libres.click()
        log("obtener_mercado: Opción 'Libres' seleccionada")

        aplicar = wait.until(
            EC.element_to_be_clickable((By.ID, "btn-filters-market-apply"))
        )
        aplicar.click()
        log("obtener_mercado: Botón 'Aplicar' clickeado")
    except Exception as e:
        log(f"obtener_mercado: Error durante el filtrado del mercado: {e}")
        return datos_mercado

    # 3. Extracción de jugadores
    try:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        lista_jugadores = soup.select("ul.player-list > li[data-owner='0']")
        log(f"obtener_mercado: {len(lista_jugadores)} jugadores encontrados en la lista")

        for jugador in lista_jugadores:
            nombre = jugador.select_one("div.info > div.name")
            nombre = nombre.text.strip() if nombre else "Desconocido"

            puntuacion = jugador.select_one("div.points")
            puntuacion = puntuacion.text.strip() if puntuacion else "0"

            precio = jugador.select_one("div.player-btns > button")
            precio = precio.text.strip() if precio else "No disponible"

            puntuacion_media = jugador.select_one("div.avg")
            puntuacion_media = puntuacion_media.text.strip() if puntuacion_media else "-"

            datos_jugador = {
                'Nombre': nombre,
                'Puntuacion': puntuacion,
                'Precio': precio,
                'Puntuacion_media': puntuacion_media,
            }

            datos_mercado.append(datos_jugador)
            log(f"obtener_mercado: Jugador añadido - {nombre} | {precio} | {puntuacion_media}")
    except Exception as e:
        log(f"obtener_mercado: Error al procesar los datos del mercado: {e}")

    log("obtener_mercado: Finalización de la función")
    return datos_mercado

