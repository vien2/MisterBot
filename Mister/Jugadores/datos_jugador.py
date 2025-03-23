from pymongo import MongoClient
import configparser
from datetime import datetime
from generate_hash import hash_dato

def insertar_datos_jugadores(temporada_actual, datos_jugador, datos_jornadas, registros_transferencia, puntos, valores):
    fecha_de_carga = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    config = configparser.ConfigParser()
    config.read('Mister\config.ini')

    client_uri = config['MongoDB']['client']
    nombre_base_datos = config['MongoDB']['nombre_base_datos']
    colecciones = config['MongoDB']['colecciones'].split(',')
    
    # Seleccionar o crear la base de datos
    client = MongoClient(client_uri)
    db = client[nombre_base_datos]

    try:
        for coleccion_nombre in colecciones:
            if coleccion_nombre == 'Datos_Jugador':
                coleccion = db[coleccion_nombre]
                for dato in datos_usuarios:
                    dato_hash = hash_dato(dato)
                    dato['F_CARGA'] = fecha_de_carga
                    dato['temporada'] = temporada_actual
                    coleccion.update_one({'hash': dato_hash}, {'$set': dato}, upsert=True)

        # Cerrar conexión a MongoDB
        client.close()

        print(f"Datos de los jugadores insertados/actualizados correctamente en la coleccion {coleccion.name}.")
    except Exception as e:
        print(f"Error al insertar datos en la colección {coleccion.name}: {e}")