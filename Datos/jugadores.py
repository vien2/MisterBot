import mysql.connector
from datetime import datetime

def insertar_jugadores(cnn, datos_jugadores, temporada):
    fecha_de_carga = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        cursor = cnn.cursor()

        for jugador_data in datos_jugadores:
            jugador = jugador_data["Jugador"]
            posicion = jugador_data["Posicion"]

            # Consultar si el jugador ya existe en la tabla Jugadores
            sql_select_jugador = "SELECT * FROM Jugadores WHERE Nombre = %s"
            cursor.execute(sql_select_jugador, (jugador,))
            existing_jugador = cursor.fetchone()

            if not existing_jugador:
                # El jugador no existe en la tabla, insertarlo
                sql_insert_jugador = "INSERT INTO Jugadores (Nombre, Posicion, Temporada, f_carga) VALUES (%s, %s, %s, %s)"
                data_insert = (jugador, posicion, temporada, fecha_de_carga)
                cursor.execute(sql_insert_jugador, data_insert)
                cnn.commit()
                print(f"Jugador {jugador} insertado correctamente.")
            else:
                # El jugador ya existe, no hacer nada
                print(f"El jugador {jugador} ya está registrado en la tabla Jugadores.")
            
    except mysql.connector.Error as error:
        print("Error al insertar datos en la tabla Jugadores:", error)
