# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import configparser, traceback, time, os

from utils import log, cerrar_popup_anuncios  # tus utilidades

# ----------------- utilidades locales -----------------
def _config_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "config.ini"),
        os.path.join(os.getcwd(), "config.ini"),
        "/home/vien2/MisterBot/Mister/config.ini",  # VPS
        r"C:\Python\MisterBot\Mister\config.ini",   # Windows local
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return candidates[0]

def _dump_state(driver, prefix):
    ts = int(time.time())
    try: driver.save_screenshot(f"{prefix}_{ts}.png")
    except: pass
    try:
        with open(f"{prefix}_{ts}.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
    except: pass

def _build_driver(headless=True):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--lang=es-ES")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
    )
    try:
        return webdriver.Chrome(options=opts)
    except WebDriverException as e:
        # fallback a headless clásico si hace falta
        if headless:
            try:
                # no hay setter de 'arguments', así que añadimos y ya
                opts.add_argument("--headless")
                return webdriver.Chrome(options=opts)
            except Exception:
                pass
        # último recurso: chromedriver_autoinstaller
        import chromedriver_autoinstaller
        chromedriver_autoinstaller.install()
        return webdriver.Chrome(options=opts)

def _robust_click(driver, locator, timeout=12):
    wait = WebDriverWait(driver, timeout)
    el = wait.until(EC.presence_of_element_located(locator))
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    try:
        wait.until(EC.element_to_be_clickable(locator)).click()
    except Exception:
        driver.execute_script("arguments[0].click();", el)

def _aceptar_cookies(driver):
    # Didomi en varias variantes
    try: _robust_click(driver, (By.ID, "didomi-notice-agree-button"), 6); return
    except: pass
    try: _robust_click(driver, (By.XPATH, "//button[contains(., 'Aceptar') or contains(., 'Agree')]"), 5); return
    except: pass
    try:
        for fr in driver.find_elements(By.CSS_SELECTOR, "iframe"):
            meta = (fr.get_attribute("id") or "") + (fr.get_attribute("name") or "") + (fr.get_attribute("src") or "")
            if "didomi" in meta.lower():
                driver.switch_to.frame(fr)
                try:
                    _robust_click(driver, (By.ID, "didomi-notice-agree-button"), 5)
                    driver.switch_to.default_content()
                    return
                except:
                    driver.switch_to.default_content()
    except:
        try: driver.switch_to.default_content()
        except: pass
# ------------------------------------------------------

def iniciar_sesion(schema=None, headless=True):
    log("iniciar_sesion: Inicio de la función")

    # Cargar config
    cfg_path = _config_path()
    config = configparser.ConfigParser()
    read_ok = bool(config.read(cfg_path))
    log(f"iniciar_sesion: usando config.ini → {cfg_path} (leído={read_ok})")

    correo = config.get('Credentials', 'username', fallback=None)
    clave  = config.get('Credentials', 'password', fallback=None)

    driver = None
    try:
        driver = _build_driver(headless=headless)
        wait = WebDriverWait(driver, 25)

        # 1) Onboarding
        log("login: GET /new-onboarding")
        driver.get("https://mister.mundodeportivo.com/new-onboarding/")

        _aceptar_cookies(driver)

        # En el VPS el botón es 'Next' con clase .btn--primary (según tu HTML dump)
        # pulsamos hasta 4 veces si aparece
        #try:
        #    for i in range(4):
        #        _robust_click(driver, (By.CSS_SELECTOR, ".btn.btn--primary"), 6)  # Next/Siguiente
        #        time.sleep(0.6)
        #except Exception:
        #    pass
        #_dump_state(driver, "onboard")

        # 2) “Continuar con Email” (ES/EN) o fallback a /login
        clicked_email = False
        for xp in [
            "//button[.//span[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continuar con email')]]",
            "//button[.//span[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue with email')]]",
            "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue with email')]",
        ]:
            try:
                _robust_click(driver, (By.XPATH, xp), 6)
                clicked_email = True
                break
            except Exception:
                continue

        if not clicked_email:
            log("login: no aparece 'Continuar/Continue with Email' → fallback a /login")
            driver.get("https://mister.mundodeportivo.com/login")

        # 3) Formulario email/password
        log("login: esperando campo email")
        #_dump_state(driver, "login_page")
        email_box = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input#email, input[name='email']"))
        )
        email_box.clear(); email_box.send_keys(correo)

        _robust_click(driver, (By.XPATH, "//button[contains(., 'Continuar') or contains(., 'Continue')]"), 12)

        pwd_box = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']"))
        )
        pwd_box.clear(); pwd_box.send_keys(clave)
        _robust_click(driver, (By.XPATH, "//button[contains(., 'Continuar') or contains(., 'Continue')]"), 12)

        # 4) App cargada
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "menu")))

        try:
            cerrar_popup_anuncios(driver)
        except Exception:
            pass

        # 5) Seleccionar liga si procede
        if schema:
            log(f"Buscando liga correspondiente al schema '{schema}'...")
            def norm(s: str) -> str:
                return s.lower().replace(" ", "").replace("_", "").replace("fc", "").strip()
            try:
                sidebar = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "sidebar")))
                liga_links = sidebar.find_elements(By.CSS_SELECTOR, ".communities ul li a")
                for link in liga_links:
                    try:
                        nombre = link.find_element(By.CLASS_NAME, "name").get_attribute("innerText").strip()
                        log(f"Nombre liga detectado: '{nombre}'")
                        if norm(schema) in norm(nombre):
                            if "active" not in (link.get_attribute("class") or ""):
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
        if driver: _dump_state(driver, "timeout")
        log(f"Timeout durante el login: {e}")
        raise
    except Exception as e:
        if driver: _dump_state(driver, "crash")
        log("Error en la ejecución: " + "".join(traceback.format_exception_only(type(e), e)).strip())
        raise
