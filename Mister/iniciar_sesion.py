from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import configparser
from selenium.webdriver.chrome.service import Service as ChromeService
from utils import log

def iniciar_sesion():
    log("iniciar_sesion: Inicio de la función")
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    log("iniciar_sesion: Configuración leída desde 'config.ini'")
    
    ruta_chromedriver = r"C:\Users\juan_\AppData\Local\chromedriver\chromedriver-win64\chromedriver.exe"
    service = ChromeService(executable_path=ruta_chromedriver)
    driver = webdriver.Chrome(service=service)
    log("iniciar_sesion: ChromeDriver iniciado")
    
    driver.get("https://mister.mundodeportivo.com/new-onboarding/")
    log("iniciar_sesion: Accediendo a la URL 'https://mister.mundodeportivo.com/new-onboarding/'")
    
    wait = WebDriverWait(driver, 10)
    
    boton_aceptar = wait.until(EC.element_to_be_clickable((By.ID, "didomi-notice-agree-button")))
    boton_aceptar.click()
    log("iniciar_sesion: Botón 'Aceptar' clickeado")
    
    boton_siguiente = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Siguiente')]")))
    boton_siguiente.click()
    log("iniciar_sesion: Botón 'Siguiente' (1) clickeado")
    
    boton_siguiente_2 = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Siguiente')]")))
    boton_siguiente_2.click()
    log("iniciar_sesion: Botón 'Siguiente' (2) clickeado")
    
    boton_siguiente_3 = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Siguiente')]")))
    boton_siguiente_3.click()
    log("iniciar_sesion: Botón 'Siguiente' (3) clickeado")
    
    boton_siguiente_4 = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Empezar')]")))
    boton_siguiente_4.click()
    log("iniciar_sesion: Botón 'Empezar' clickeado")
    
    boton_siguiente_5 = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[.//span[contains(text(), 'Continuar con Email')]]")))
    boton_siguiente_5.click()
    log("iniciar_sesion: Botón 'Continuar con Email' clickeado")
    
    campo_correo = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "email")))
    log("iniciar_sesion: Campo de correo electrónico localizado")
    campo_correo.clear()
    
    config = configparser.ConfigParser()
    config.read("C:\\Python\\Mister-bot\\Datos\\config.ini")
    log("iniciar_sesion: Configuración leída desde 'C:\\Python\\Mister-bot\\Datos\\config.ini'")
    
    correo_electronico = config['Credentials']['username']
    contraseña = config['Credentials']['password']
    campo_correo.send_keys(correo_electronico)
    log("iniciar_sesion: Correo electrónico ingresado")
    
    boton_siguiente_6 = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), ' Continuar')]")))
    boton_siguiente_6.click()
    log("iniciar_sesion: Botón 'Continuar' (correo) clickeado")
    
    campo_contraseña = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']")))
    campo_contraseña.clear()
    campo_contraseña.send_keys(contraseña)
    log("iniciar_sesion: Contraseña ingresada")
    
    boton_siguiente_7 = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), ' Continuar')]")))
    boton_siguiente_7.click()
    log("iniciar_sesion: Botón 'Continuar' (contraseña) clickeado")
    
    log("iniciar_sesion: Finalización exitosa de la función, retornando driver")
    return driver
