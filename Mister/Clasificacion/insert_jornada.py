from pymongo import MongoClient
from datetime import datetime
import configparser
import ast
from generate_hash import hash_dato

def insertar_datos_clasificacion_jornada(temporada_actual, datos_por_jornada):
  fecha_de_carga = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

  config = configparser.ConfigParser()
  config.read('Mister\config.ini')

  client_uri = config['MongoDB']['client']
  nombre_base_datos = config['MongoDB']['nombre_base_datos']
  colecciones = config['MongoDB']['colecciones'].split(',')

  # Conectar a MongoDB
  client = MongoClient(client_uri)
  db = client[nombre_base_datos]

  coleccion = None
  try:
    # Insertar datos en las colecciones especificadas
    for coleccion_nombre in colecciones:
      if coleccion_nombre == 'Clasificacion_jornada':
        coleccion = db[coleccion_nombre]
        # Insertar datos en la colección
        for jornada, datos_jornadas in datos_por_jornada.items():
          for dato in datos_jornadas:
            if isinstance(dato, str):
              dato = ast.literal_eval(dato)
            dato_hash = hash_dato(dato)
            dato['F_CARGA'] = fecha_de_carga
            dato['temporada'] = temporada_actual
            coleccion.update_one({'hash': dato_hash}, {'$set': dato}, upsert=True)

    # Cerrar conexión a MongoDB
    client.close()

    print(f"Datos insertados correctamente en la colección {coleccion.name}")
  except Exception as e:
    print(f"Error al insertar datos en la colección {coleccion.name}: {e}")
