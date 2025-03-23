from Robo.robo_jugador import robo_jugador
from selenium import webdriver
from bs4 import BeautifulSoup
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait,Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
import json, re, csv
import configparser
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

def iniciar_sesion():

  config = configparser.ConfigParser()
  config.read('config.ini')
  
  ruta_chromedriver = r"C:\Users\juan_\AppData\Local\chromedriver\chromedriver-win64\chromedriver.exe"
  service = ChromeService(executable_path=ruta_chromedriver)
  driver = webdriver.Chrome(service=service)
  driver.get("https://mister.mundodeportivo.com/new-onboarding/")

  wait = WebDriverWait(driver, 10)
  boton_aceptar = wait.until(EC.element_to_be_clickable((By.ID, "didomi-notice-agree-button")))
  boton_aceptar.click()
  
  boton_siguiente = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Siguiente')]")))
  boton_siguiente.click()
  
  boton_siguiente_2 = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Siguiente')]")))
  boton_siguiente_2.click()
  
  boton_siguiente_3 = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Siguiente')]")))
  boton_siguiente_3.click()
  
  boton_siguiente_4 = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Empezar')]")))
  boton_siguiente_4.click()
  #time.sleep(5000)
  boton_siguiente_5 = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[.//span[contains(text(), 'Continuar con Email')]]")))
  boton_siguiente_5.click()
  #time.sleep(5000)
  # Localizar los campos de correo electrónico y contraseña
  campo_correo = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "email")))

  campo_correo.clear()

  config = configparser.ConfigParser()
  config.read("C:\Python\Mister-bot\Datos\config.ini")

  # Escribir el correo electrónico y la contraseña
  correo_electronico = config['Credentials']['username']
  contraseña = config['Credentials']['password']
  campo_correo.send_keys(correo_electronico)

  boton_siguiente_6 = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), ' Continuar')]")))
  boton_siguiente_6.click()
  campo_contraseña = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']")))
  campo_contraseña.clear()
  campo_contraseña.send_keys(contraseña)

  boton_siguiente_7 = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), ' Continuar')]")))
  boton_siguiente_7.click()

  return driver