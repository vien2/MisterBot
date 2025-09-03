from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from utils import log
import time
import re
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException

def _normalizar_jornada_texto(txt: str) -> str:
    """
    Convierte 'J1' -> 'Jornada 1', 'J10' -> 'Jornada 10'.
    Si ya viene como 'Jornada X', lo deja igual.
    """
    t = (txt or "").strip()
    if t.lower().startswith("jornada"):
        return t
    m = re.match(r"j\s*([0-9]+)", t, re.IGNORECASE)
    return f"Jornada {m.group(1)}" if m else t

def _url_absoluta(url_rel_o_abs: str) -> str:
    if not url_rel_o_abs:
        return url_rel_o_abs
    if url_rel_o_abs.startswith("http"):
        return url_rel_o_abs
    # asegura el slash inicial
    if not url_rel_o_abs.startswith("/"):
        url_rel_o_abs = "/" + url_rel_o_abs
    return "https://mister.mundodeportivo.com" + url_rel_o_abs

def obtener_best_xi_jornadas_finalizadas(driver, schema=None):
    _ = schema
    log("obtener_best_xi_jornadas_finalizadas: Inicio")
    datos = []
    wait = WebDriverWait(driver, 10)

    # 1) Abrir el selector de jornadas (icono flecha en feed-top-gameweek)
    try:
        # Si ya está abierto, este click no hace daño; si no lo está, lo abre.
        icono = wait.until(
            EC.element_to_be_clickable(
                (By.XPATH, "//div[@class='feed-top-gameweek']//div[@class='gameweek__icon']")
            )
        )
        icono.click()
        log("obtener_best_xi_jornadas_finalizadas: Icono de jornadas clickeado")
        # Pequeña pausa para que aparezca el selector
        time.sleep(0.5)
    except Exception as e:
        log(f"obtener_best_xi_jornadas_finalizadas: Error al hacer clic en el icono: {e}")
        return []

    # 2) Iterar por todas las jornadas del selector
    try:
        # Contar cuántos botones de jornada hay
        botones_iniciales = wait.until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.gameweek-selector-inline button"))
        )
        total_jornadas = len(botones_iniciales)
        log(f"obtener_best_xi_jornadas_finalizadas: {total_jornadas} jornadas encontradas")
    except TimeoutException:
        log("obtener_best_xi_jornadas_finalizadas: No se encontró el selector de jornadas")
        return []

    for idx in range(total_jornadas):
        try:
            # Releer el botón en cada iteración para evitar elementos obsoletos
            botones = driver.find_elements(By.CSS_SELECTOR, "div.gameweek-selector-inline button")
            if idx >= len(botones):
                break  # por seguridad si el DOM cambió
            btn = botones[idx]
            jornada_texto_raw = btn.text.strip()
            jornada_texto = _normalizar_jornada_texto(jornada_texto_raw)
            jornada_id = btn.get_attribute("data-id") or ""

            # Click en la jornada
            try:
                # intenta click normal
                btn.click()
            except StaleElementReferenceException:
                # reintento rápido si quedó obsoleto
                botones = driver.find_elements(By.CSS_SELECTOR, "div.gameweek-selector-inline button")
                btn = botones[idx]
                btn.click()

            log(f"obtener_best_xi_jornadas_finalizadas: Jornada '{jornada_texto}' seleccionada (id {jornada_id})")

            # Esperar a que cargue algo de contenido de la jornada
            time.sleep(0.8)

            # 3) Confirmar que la jornada está FINALIZADA
            try:
                # Busca el título "Finalizada" en una section-title
                WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located(
                        (By.XPATH, "//div[@class='section-title']/h3[normalize-space()='Finalizada']")
                    )
                )
            except TimeoutException:
                # No finalizada → saltar
                log(f"obtener_best_xi_jornadas_finalizadas: Jornada '{jornada_texto}' no finalizada. Se omite.")
                continue

            # 4) Extraer el Best XI de esta jornada
            try:
                jugadores = driver.find_elements(
                    By.CSS_SELECTOR, ".team-lineup .lineup-best-xi a.lineup-player.best-xi"
                )
            except Exception as e:
                log(f"obtener_best_xi_jornadas_finalizadas: No se encontraron jugadores en {jornada_texto}: {e}")
                continue

            if not jugadores:
                log(f"obtener_best_xi_jornadas_finalizadas: No hay jugadores en {jornada_texto}")
                continue

            for a in jugadores:
                try:
                    url = _url_absoluta(a.get_attribute("href"))
                    id_jugador = a.get_attribute("data-id_player")
                    nombre = a.find_element(By.CSS_SELECTOR, ".info .name").text.strip()
                    puntos = a.find_element(By.CSS_SELECTOR, ".info .points").text.strip()

                    datos.append({
                        "id_jugador": id_jugador,
                        "nombre": nombre,
                        "url": url,
                        "puntos": puntos,
                        "jornada": jornada_texto
                    })
                    log(f"obtener_best_xi_jornadas_finalizadas: {nombre} ({id_jugador}) - {puntos} puntos en {jornada_texto}")
                except Exception as e:
                    log(f"obtener_best_xi_jornadas_finalizadas: Error procesando jugador en '{jornada_texto}': {e}")
                    continue

        except Exception as e:
            log(f"obtener_best_xi_jornadas_finalizadas: Error en jornada índice {idx}: {e}")
            continue

    log(f"obtener_best_xi_jornadas_finalizadas: Total registros capturados: {len(datos)}")
    return datos