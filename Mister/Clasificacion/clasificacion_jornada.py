from datetime import datetime
import mysql.connector

def insertar_datos_jornada(cnn, datos_jornada):
    # Obtener la fecha de carga actual en formato datetime
    fecha_de_carga = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        cursor = cnn.cursor()

        for usuario_data in datos_jornada:
            jornada = usuario_data["Jornada"]
            usuario = usuario_data["Nombre"]
            temporada = "23/24"
            posicion = usuario_data["Posicion"]
            puntos = usuario_data["Puntos"]
            
            # Obtener el ID del usuario desde la tabla "Usuarios"
            cursor.execute("SELECT ID FROM Usuarios WHERE Nombre = %s", (usuario,))
            usuario_id = cursor.fetchone()

            if usuario_id:
                usuario_id = usuario_id[0]
                # Comprobar si el usuario ya tiene un registro para esta jornada
                cursor.execute("SELECT ID FROM Clasificacion_jornada WHERE UsuarioID = %s AND Jornada = %s",
                               (usuario_id, jornada))
                existing_jornada = cursor.fetchone()

                if not existing_jornada:
                    # Insertar un nuevo registro para este usuario y jornada
                    sql_insert_user_jornada = ("INSERT INTO Clasificacion_jornada (UsuarioID, Temporada, Jornada, Posicion, Puntos, f_carga) "
                                               "VALUES (%s,%s, %s, %s, %s, %s)")
                    data_insert = (usuario_id, temporada, jornada, posicion, puntos, fecha_de_carga)
                    cursor.execute(sql_insert_user_jornada, data_insert)
                    cnn.commit()
                    print(f"Registro de {usuario} en la temporada {temporada} para la jornada {jornada} insertado.")
                else:
                    print(f"El usuario {usuario} ya tiene un registro para la jornada {jornada} de la temporada {temporada}. No se realizará la inserción.")
            else:
                print(f"El usuario {usuario} no existe en la tabla de Usuarios.")
                
    except mysql.connector.Error as error:
        print("Error al insertar datos en la tabla Clasificacion_jornada:", error)
