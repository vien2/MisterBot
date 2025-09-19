import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from utils import log


def poner_en_venta_equipo(driver, schema=None):
    log("poner_en_venta_equipo: Inicio del proceso")

    wait = WebDriverWait(driver, 10)

    # --- Ir a la pestaña Equipo ---
    try:
        enlace_equipo = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//ul[@class='menu']//li[@data-pag='team']/a")
        ))
        enlace_equipo.click()
        log("poner_en_venta_equipo: Enlace 'Equipo' clickeado")
    except Exception as e:
        log(f"poner_en_venta_equipo: Error al ir a 'Equipo': {e}")
        return

    # --- Esperar a que cargue la lista de jugadores ---
    try:
        wait.until(EC.presence_of_all_elements_located(
            (By.XPATH, "//ul[contains(@class,'list-team')]/li")
        ))
    except Exception as e:
        log(f"poner_en_venta_equipo: No se detectó la lista de jugadores: {e}")
        return

    # --- Obtener todos los botones de jugadores ---
    jugadores = driver.find_elements(By.XPATH, "//ul[contains(@class,'list-team')]/li")

    log(f"poner_en_venta_equipo: {len(jugadores)} jugadores encontrados")

    for jugador in jugadores:
        try:
            boton = jugador.find_element(By.XPATH, ".//div[@class='player-btns']/button")
            texto_boton = boton.text.strip()

            if texto_boton.lower() == "gestionar":
                log("poner_en_venta_equipo: Jugador no está en venta → clic en Gestionar")
                boton.click()

                # Esperar a que se abra el popup
                try:
                    boton_vender = wait.until(EC.element_to_be_clickable(
                        (By.XPATH, "//div[@id='popup']//button[@id='btn-send' and @data-ajax='sale']")
                    ))
                    time.sleep(0.2)
                    boton_vender.click()
                    log("poner_en_venta_equipo: Botón 'Vender' clickeado")

                    # Esperar a que el popup desaparezca
                    wait.until(EC.invisibility_of_element_located((By.ID, "popup")))
                    log("poner_en_venta_equipo: Popup cerrado, jugador puesto en venta")
                except Exception as e:
                    log(f"poner_en_venta_equipo: Error al vender jugador: {e}")
                    continue

            else:
                log("poner_en_venta_equipo: Jugador ya estaba en venta → no hago nada")

        except Exception as e:
            log(f"poner_en_venta_equipo: Error procesando jugador: {e}")
            continue

    log("poner_en_venta_equipo: Todos los jugadores procesados correctamente")

def poner_mercado_todo(driver, schema=None):
    _ = schema
    log("poner_mercado_todo: Inicio del proceso")
    wait = WebDriverWait(driver, 15)

    # 1) Ir a la pestaña "Mercado"
    try:
        enlace_mercado = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//ul[@class='menu']//li[@data-pag='market']/a")
        ))
        enlace_mercado.click()
        log("poner_mercado_todo: Enlace 'Mercado' clickeado")
    except Exception as e:
        log(f"poner_mercado_todo: No se pudo clickar 'Mercado': {e}")
        return

    # 2) Abrir sección ofertas recibidas
    try:
        btn_ofertas = wait.until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR, "div.btn.btn-sw.offers.offers-received")
        ))
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn_ofertas)
        time.sleep(0.2)
        btn_ofertas.click()
        log("poner_mercado_todo: Botón 'Ofertas recibidas' clickeado")
    except Exception as e:
        log(f"poner_mercado_todo: No se pudo abrir ofertas recibidas: {e}")
        return

    # 3) Esperar a que cargue la lista de jugadores con ofertas
    players_xpath = "//ul[contains(@class,'sw-market-offers')]/li[contains(@class,'offer-wrapper')]"
    try:
        wait.until(EC.presence_of_all_elements_located((By.XPATH, players_xpath)))
    except Exception:
        log("poner_mercado_todo: No quedan jugadores con ofertas → fin")
        return

    # 4) Procesar jugadores en bucle
    while True:
        jugadores = driver.find_elements(By.XPATH, players_xpath)
        if not jugadores:
            log("poner_mercado_todo: No quedan jugadores con ofertas → fin")
            break

        jugador = jugadores[0]  # siempre el primero, porque desaparece al procesar
        try:
            nombre = jugador.find_element(By.CSS_SELECTOR, ".info .name").text.strip()
        except:
            nombre = "desconocido"

        try:
            # Clic en botón revender
            btn_revender = jugador.find_element(By.CSS_SELECTOR, "button.btn-resale")
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn_revender)
            time.sleep(0.2)
            btn_revender.click()
            log(f"poner_mercado_todo: Clic en revender → {nombre}")

            # Pop-up → "Sí, seguro"
            btn_confirmar = wait.until(EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "div.popup button#btn-send[data-ajax='resale']")
            ))
            btn_confirmar.click()
            log(f"poner_mercado_todo: Confirmado revender → {nombre}")

            # Esperar a que el jugador desaparezca
            wait.until(EC.staleness_of(jugador))
            log(f"poner_mercado_todo: Jugador eliminado de la lista → {nombre}")
            time.sleep(0.5)

        except Exception as e:
            log(f"poner_mercado_todo: Error con jugador {nombre}: {e}")
            break

    log("poner_mercado_todo: Todos los jugadores procesados")

