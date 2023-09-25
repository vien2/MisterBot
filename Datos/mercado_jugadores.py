import mysql.connector
from datetime import datetime

def insertar_datos_mercado(cnn, datos_mercado, temporada):
    fecha_de_carga = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        cursor = cnn.cursor()

        for jugador_data in datos_mercado:
            nombre = jugador_data["Nombre"]
            puntuacion = jugador_data["Puntuacion"]
            precio = jugador_data["Precio"]
            puntuacion_media = jugador_data["Puntuacion_media"]

            # Consultar si el jugador ya existe en la tabla Jugadores
            sql_select_jugador = "SELECT ID FROM Jugadores WHERE Nombre = %s"
            cursor.execute(sql_select_jugador, (nombre,))
            jugador_id = cursor.fetchone()

            if jugador_id:
                # El jugador ya existe en la tabla Jugadores, insertar en Mercado_jugadores
                sql_insert_mercado = ("INSERT INTO mercado_jugadores "
                                      "(JugadorID, Puntuacion, Precio, PuntuacionMedia, Temporada, f_carga) "
                                      "VALUES (%s, %s, %s, %s, %s, %s)")
                data_insert = (jugador_id[0], puntuacion, precio, puntuacion_media, temporada, fecha_de_carga)
                cursor.execute(sql_insert_mercado, data_insert)
                cnn.commit()
                print(f"Jugador {nombre} insertado correctamente en el mercado.")
            else:
                # El jugador no existe en la tabla Jugadores, mostrar mensaje de error
                print(f"El jugador {nombre} no existe en la tabla Jugadores.")
            
    except mysql.connector.Error as error:
        print("Error al insertar datos en la tabla Mercado_jugadores:", error)