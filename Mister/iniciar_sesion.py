from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import configparser
from selenium.webdriver.chrome.service import Service as ChromeService
from utils import log

def iniciar_sesion(schema=None):
    log("iniciar_sesion: Inicio de la función")

    config = configparser.ConfigParser()
    config.read("C:\\Python\\Mister-bot\\Datos\\config.ini")
    ruta_chromedriver = r"C:\Users\juan_\AppData\Local\chromedriver\chromedriver-win64\chromedriver.exe"
    service = ChromeService(executable_path=ruta_chromedriver)
    driver = webdriver.Chrome(service=service)

    driver.get("https://mister.mundodeportivo.com/new-onboarding/")
    wait = WebDriverWait(driver, 10)

    # Aceptar cookies
    wait.until(EC.element_to_be_clickable((By.ID, "didomi-notice-agree-button"))).click()

    # Avanzar pantallas introductorias
    for texto in ["Siguiente", "Siguiente", "Siguiente", "Empezar"]:
        boton = wait.until(EC.element_to_be_clickable((By.XPATH, f"//button[contains(text(), '{texto}')]")))
        boton.click()

    # Continuar con email
    wait.until(EC.element_to_be_clickable((By.XPATH, "//button[.//span[contains(text(), 'Continuar con Email')]]"))).click()

    # Login
    correo_electronico = config['Credentials']['username']
    contraseña = config['Credentials']['password']
    wait.until(EC.presence_of_element_located((By.ID, "email"))).send_keys(correo_electronico)
    wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), ' Continuar')]"))).click()

    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']"))).send_keys(contraseña)
    wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), ' Continuar')]"))).click()

    # Esperamos a que cargue el menú principal
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "menu")))

    # Seleccionar liga si se especificó un schema
    if schema:
        log(f"Buscando liga correspondiente al schema '{schema}'...")

        def normalizar(s):
            return s.lower().replace(" ", "").replace("_", "").replace("fc", "").strip()

        try:
            sidebar = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "sidebar"))
            )
            liga_links = sidebar.find_elements(By.CSS_SELECTOR, ".communities ul li a")

            for link in liga_links:
                try:
                    name_elem = link.find_element(By.CLASS_NAME, "name")
                    nombre_liga = name_elem.get_attribute("innerText").strip()
                    log(f"Nombre liga detectado: '{nombre_liga}'")

                    if normalizar(schema) in normalizar(nombre_liga):
                        log(f"Liga encontrada: '{nombre_liga}'")

                        clases = link.get_attribute("class") or ""
                        if "active" in clases:
                            log(f"La liga '{nombre_liga}' ya está activa, no es necesario hacer clic.")
                        else:
                            log(f"Haciendo clic en la liga '{nombre_liga}'...")
                            try:
                                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link)
                                WebDriverWait(driver, 5).until(EC.element_to_be_clickable(link))
                                link.click()
                            except Exception as e_click:
                                log(f"Error estándar al hacer clic: {e_click}")
                                log("Intentando con JavaScript...")
                                try:
                                    driver.execute_script("arguments[0].click();", link)
                                except Exception as js_error:
                                    log(f"Error también con JavaScript click: {js_error}")
                        break
                except Exception as e_liga:
                    log(f"Error leyendo nombre de liga: {e_liga}")
            else:
                log(f"No se encontró liga correspondiente a '{schema}'")
        except Exception as e:
            log(f"Error general al intentar seleccionar liga: {e}")

    log("Sesión iniciada y liga seleccionada (si corresponde)")
    return driver
