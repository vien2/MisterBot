# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import configparser
import traceback
import time

# Tus utilidades
from utils import log, cerrar_popup_anuncios


# ---------- Helpers ----------
def _build_driver(headless=True):
    """
    Crea el driver de Chrome usando Selenium Manager (preferido).
    Si falla con --headless=new, hace fallback automático a --headless clásico.
    Como último recurso intenta con chromedriver_autoinstaller.
    """
    opts = Options()

    # Headless + flags útiles para entornos sin GPU/CI
    if headless:
        opts.add_argument("--headless=new")          # fallback más abajo si no está soportado
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")

    # Opciones “cosméticas” que ayudan en ciertos sitios
    opts.add_argument("--lang=es-ES")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
    )

    try:
        # 1) Selenium Manager (no requiere chromedriver instalado)
        return webdriver.Chrome(options=opts)
    except WebDriverException:
        # 2) Fallback: headless clásico
        if headless:
            try:
                # Quita --headless=new si estaba y usa --headless clásico
                opts.arguments = [a for a in opts.arguments if not a.startswith("--headless")]
                opts.add_argument("--headless")
                return webdriver.Chrome(options=opts)
            except Exception:
                pass

        # 3) Último recurso: chromedriver_autoinstaller
        try:
            import chromedriver_autoinstaller
            chromedriver_autoinstaller.install()
            return webdriver.Chrome(options=opts)
        except Exception as e:
            raise e


def _dump_state(driver, prefix="fallo"):
    """
    Guarda screenshot y HTML del estado actual para depurar.
    """
    ts = int(time.time())
    try:
        driver.save_screenshot(f"{prefix}_{ts}.png")
    except Exception:
        pass
    try:
        with open(f"{prefix}_{ts}.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
    except Exception:
        pass


def _robust_click(driver, locator, timeout=10):
    """
    Click robusto: asegura presencia, hace scroll al centro, intenta click Selenium
    y si falla, click por JavaScript.
    """
    wait = WebDriverWait(driver, timeout)
    el = wait.until(EC.presence_of_element_located(locator))
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    try:
        wait.until(EC.element_to_be_clickable(locator)).click()
    except Exception:
        driver.execute_script("arguments[0].click();", el)


def _aceptar_cookies(driver):
    """
    Acepta cookies probando varios selectores y, si hace falta, dentro de iframes de didomi.
    No lanza excepción si no encuentra nada (algunas regiones no muestran banner).
    """
    # 1) ID clásico de Didomi
    try:
        _robust_click(driver, (By.ID, "didomi-notice-agree-button"), 8)
        return
    except Exception:
        pass

    # 2) Por texto (ES/EN)
    try:
        _robust_click(driver, (By.XPATH, "//button[contains(., 'Aceptar') or contains(., 'Agree')]"), 6)
        return
    except Exception:
        pass

    # 3) Dentro de iframes con "didomi" en id/name/src
    try:
        iframes = driver.find_elements(By.CSS_SELECTOR, "iframe")
        for fr in iframes:
            meta = (fr.get_attribute("id") or "") + (fr.get_attribute("name") or "") + (fr.get_attribute("src") or "")
            if "didomi" in meta.lower():
                driver.switch_to.frame(fr)
                try:
                    _robust_click(driver, (By.ID, "didomi-notice-agree-button"), 5)
                    driver.switch_to.default_content()
                    return
                except Exception:
                    driver.switch_to.default_content()
    except Exception:
        try:
            driver.switch_to.default_content()
        except Exception:
            pass
    # Si no se pudo, seguimos sin bloquear


# ---------- Función principal ----------
def iniciar_sesion(schema=None, headless=True):
    """
    Abre sesión en Mister (headless por defecto), cierra popups y selecciona liga si se pasa schema.
    Devuelve un driver listo para reutilizar.
    """
    log("iniciar_sesion: Inicio de la función")

    # Config
    config = configparser.ConfigParser()
    config.read(r"C:\Python\MisterBot\Mister\config.ini")

    driver = None
    try:
        driver = _build_driver(headless=headless)

        # Navega a onboarding/login
        driver.get("https://mister.mundodeportivo.com/new-onboarding/")
        wait = WebDriverWait(driver, 20)

        # Cookies (robusto)
        _aceptar_cookies(driver)

        # Avanzar pantallas introductorias
        for texto in ["Siguiente", "Siguiente", "Siguiente", "Empezar"]:
            _robust_click(driver, (By.XPATH, f"//button[contains(normalize-space(.), '{texto}')]"), 12)

        # Continuar con email
        _robust_click(driver, (By.XPATH, "//button[.//span[contains(., 'Continuar con Email')]]"), 12)

        # Login
        correo = config['Credentials']['username']
        clave = config['Credentials']['password']
        wait.until(EC.presence_of_element_located((By.ID, "email"))).send_keys(correo)
        _robust_click(driver, (By.XPATH, "//button[contains(normalize-space(.), 'Continuar')]"), 12)

        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']"))).send_keys(clave)
        _robust_click(driver, (By.XPATH, "//button[contains(normalize-space(.), 'Continuar')]"), 12)

        # Menú principal cargado
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "menu")))

        # Popup inicial (si lo hay)
        try:
            cerrar_popup_anuncios(driver)
        except Exception:
            pass

        # Seleccionar liga si se especificó schema
        if schema:
            log(f"Buscando liga correspondiente al schema '{schema}'...")

            def normalizar(s: str) -> str:
                return s.lower().replace(" ", "").replace("_", "").replace("fc", "").strip()

            try:
                sidebar = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "sidebar")))
                liga_links = sidebar.find_elements(By.CSS_SELECTOR, ".communities ul li a")

                for link in liga_links:
                    try:
                        nombre = link.find_element(By.CLASS_NAME, "name").get_attribute("innerText").strip()
                        log(f"Nombre liga detectado: '{nombre}'")
                        if normalizar(schema) in normalizar(nombre):
                            clases = link.get_attribute("class") or ""
                            if "active" not in clases:
                                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", link)
                                try:
                                    WebDriverWait(driver, 5).until(EC.element_to_be_clickable(link)).click()
                                except Exception:
                                    driver.execute_script("arguments[0].click();", link)
                            break
                    except Exception as e_liga:
                        log(f"Error leyendo nombre de liga: {e_liga}")
                else:
                    log(f"No se encontró liga correspondiente a '{schema}'")
            except Exception as e:
                log(f"Error general al intentar seleccionar liga: {e}")

        log("Sesión iniciada y liga seleccionada (si corresponde)")
        return driver

    except TimeoutException as e:
        if driver:
            _dump_state(driver, "timeout")
        log(f"Timeout durante el login: {e}")
        raise

    except Exception as e:
        if driver:
            _dump_state(driver, "crash")
        log("Error en la ejecución: " + "".join(traceback.format_exception_only(type(e), e)).strip())
        raise
