from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils import log

def obtener_mercado(driver):
    log("obtener_mercado: Inicio de la función")

    datos_mercado = []
    wait = WebDriverWait(driver, 10)

    try:
        enlace_mercado = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@class='header-menu']//div[contains(text(), 'Mercado')]/parent::li/a")))
        enlace_mercado.click()
        log("obtener_mercado: Enlace 'Mercado' clickeado")
    except Exception as e:
        log(f"obtener_mercado: Error al hacer clic en 'Mercado': {e}")
        return datos_mercado

    try:
        filtrar = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Filtrar')]")))
        filtrar.click()
        log("obtener_mercado: Botón 'Filtrar' clickeado")

        libres = wait.until(EC.element_to_be_clickable((By.XPATH, "//label[contains(text(), 'Libres')]")))
        libres.click()
        log("obtener_mercado: Opción 'Libres' seleccionada")

        aplicar = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Aplicar')]")))
        aplicar.click()
        log("obtener_mercado: Botón 'Aplicar' clickeado")
    except Exception as e:
        log(f"obtener_mercado: Error durante el filtrado del mercado: {e}")
        return datos_mercado

    try:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        lista_jugadores = soup.select('ul.player-list.list-sale.search-players-list > li')
        log(f"obtener_mercado: {len(lista_jugadores)} jugadores encontrados en la lista")

        for jugador in lista_jugadores:
            estilo_jugador = jugador.get('style', '')
            if "display: none" in estilo_jugador:
                continue

            nombre = jugador.select_one('div.info > div.name').text.strip()
            puntuacion = jugador.select_one('div.points').text.strip()
            precio = jugador.select_one('button.btn-popup.btn-bid.btn-green, button.btn-popup.btn-bid.btn-grey')
            if precio:
                precio = precio.text.strip()
            else:
                precio = "No disponible"

            puntuacion_media = jugador.select_one('div.avg').text.strip()

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
