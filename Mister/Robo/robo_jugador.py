from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from datetime import datetime
import time
import unicodedata
from utils import log

try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo


def _esperar_hasta_hora_madrid(hora_str: str) -> bool:
    """
    Espera (bloqueante) hasta la hora HH:mm:ss de hoy (Europe/Madrid).
    Si ya pasó, devuelve False (no ejecutar).
    """
    if not hora_str:
        log("hora_operacion vacía → no se espera"); return False

    tz = ZoneInfo("Europe/Madrid")
    now = datetime.now(tz)
    try:
        h, m, s = map(int, hora_str.split(":"))
    except Exception:
        log(f"hora_operacion inválida: '{hora_str}' → no se espera"); return False

    target = now.replace(hour=h, minute=m, second=s, microsecond=0)
    if now > target:
        log(f"hora_operacion ({hora_str} Madrid) ya pasó hoy → NO ejecutar."); return False

    while True:
        now = datetime.now(tz)
        remaining = (target - now).total_seconds()
        if remaining <= 0:
            break
        chunk = remaining if remaining < 30 else 30
        log(f"Esperando hasta {target.strftime('%H:%M:%S')} Europe/Madrid (faltan {int(remaining)}s)…")
        time.sleep(chunk)
    return True


def _robust_click(driver, el, timeout=4):
    try:
        WebDriverWait(driver, timeout).until(EC.element_to_be_clickable(el)).click()
    except Exception:
        driver.execute_script("arguments[0].click();", el)


def _norm_txt(s: str) -> str:
    s = (s or "").strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    return s


def robo_jugador(
    driver,
    schema: str,
    usuario: str = None,              # ya no se usa en este flujo, lo mantenemos por contrato
    nombre_jugador: str = None,       # admite abreviado como respaldo
    nombre_completo: str = None,      # ← NUEVO: “Matteo Darmian”
    hora_operacion: str = None,       # HH:mm:ss Madrid
    timeout: int = 8,
    **_
):
    """
    Flujo con búsqueda por nombre completo:
      1) Pestaña 'Buscar'
      2) Escribir nombre_completo en input
      3) Click en el jugador exacto del listado
      4) Botón 'Pagar cláusula' (abre popup)
      5) Esperar hasta hora_operacion (Europe/Madrid)
      6) Confirmar #btn-send en el instante
    Restricción: si la hora ya pasó hoy, retorna "SKIP_PASADA" sin ejecutar.
    """
    objetivo = (nombre_completo or nombre_jugador or "").strip()
    if not objetivo or not hora_operacion:
        log("Faltan parámetros: 'nombre_completo/nombre_jugador' y/o 'hora_operacion'.")
        return False

    wait = WebDriverWait(driver, timeout)
    objetivo_norm = _norm_txt(objetivo)

    # 1) Ir a pestaña BUSCAR
    try:
        tab_search = wait.until(EC.presence_of_element_located(
            (By.XPATH, "//li[contains(@class,'btn')][@data-pag='search']"))
        )
        a = tab_search.find_element(By.CSS_SELECTOR, "a.navbar-switch-tab")
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", a)
        _robust_click(driver, a, timeout=4)
    except TimeoutException:
        log("No se encontró pestaña 'Buscar'.")
        return False

    # 2) Input de búsqueda y escribir nombre completo
    try:
        box = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "input.search-players-input[name='search']"))
        )
        box.clear()
        box.send_keys(objetivo)
        # Pequeña espera para que carguen resultados
        time.sleep(0.8)
    except TimeoutException:
        log("No se encontró el input de búsqueda de jugadores.")
        return False

    # 3) Listado de resultados y click en el jugador que coincida por nombre (normalizado)
    try:
        # El listado típico: ul.player-list .player-row a.player
        items = wait.until(EC.presence_of_all_elements_located(
            (By.XPATH, "//ul[contains(@class,'player-list')]//a[contains(@class,'player')]"))
        )
    except TimeoutException:
        log(f"No se encontraron resultados para '{objetivo}'.")
        return False

    elegido = None
    for a in items:
        try:
            name_div = a.find_element(By.XPATH, ".//div[contains(@class,'name')]")
            nombre_res = name_div.text.strip()
            if _norm_txt(nombre_res) == objetivo_norm:
                elegido = a
                break
        except NoSuchElementException:
            continue

    if not elegido:
        # Si no hay match exacto, intenta primer resultado como fallback (opcional)
        a0 = items[0]
        try:
            name0 = a0.find_element(By.XPATH, ".//div[contains(@class,'name')]").text.strip()
            log(f"No hubo match exacto; haciendo click en el primer resultado: '{name0}'.")
        except Exception:
            pass
        elegido = a0

    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", elegido)
    _robust_click(driver, elegido, timeout=4)

    # 4) Botón 'Pagar cláusula'
    try:
        pagar_btn = WebDriverWait(driver, timeout).until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(@class,'btn-clause') and @data-popup='clause-pay']"))
        )
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", pagar_btn)
        _robust_click(driver, pagar_btn, timeout=4)
    except TimeoutException:
        log("No aparece el botón 'Pagar cláusula'.")
        return False

    # 5) Esperar hasta hora objetivo (si ya pasó → SKIP)
    if not _esperar_hasta_hora_madrid(hora_operacion):
        return "SKIP_PASADA"

    # 6) Confirmar (#btn-send) justo en la hora
    try:
        confirmar = WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.ID, "btn-send")))
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", confirmar)
        _robust_click(driver, confirmar, timeout=4)
    except TimeoutException:
        log("No se pudo confirmar (no aparece #btn-send).")
        return False

    log(f"Robo completado sobre '{objetivo}'.")
    return True
