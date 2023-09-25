import mysql.connector
from datetime import datetime

def insertar_datos_clasificacion_general(cnn, datos_usuarios):
    fecha_de_carga = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Temporada actual (estática)
    temporada_actual = "23/24"
    
    try:
        cursor = cnn.cursor()

        for usuario_data in datos_usuarios:
            usuario = usuario_data["Usuario"]
            posicion = usuario_data["Posicion"]
            puntos = usuario_data["Puntos"]
            jugadores = usuario_data["Jugadores"]
            valor_total = usuario_data["Valor total"]

            # Consultar el ID del usuario en la tabla Usuarios
            sql_select_user_id = "SELECT ID FROM Usuarios WHERE Nombre = %s"
            cursor.execute(sql_select_user_id, (usuario,))
            user_id = cursor.fetchone()

            if user_id:
                # Consultar si el usuario ya tiene un registro en Clasificacion_general
                sql_select_user_clasif = "SELECT * FROM Clasificacion_general WHERE UsuarioID = %s"
                cursor.execute(sql_select_user_clasif, (user_id[0],))
                existing_user_clasif = cursor.fetchone()

                if existing_user_clasif:
                    # El usuario ya tiene un registro en Clasificacion_general
                    # Verificar si los datos han cambiado
                    if (existing_user_clasif[3] != posicion or existing_user_clasif[4] != puntos or existing_user_clasif[5] != jugadores or
                    existing_user_clasif[6] != valor_total or existing_user_clasif[7] != temporada_actual):
                        # Los datos han cambiado o la temporada ha cambiado, actualizar el registro
                        sql_update_user_clasif = ("UPDATE Clasificacion_general "
                                                  "SET Posicion = %s, Puntos = %s, Jugadores = %s, ValorTotal = %s, Temporada = %s, f_carga = %s "
                                                  "WHERE UsuarioID = %s")
                        data_update = (posicion, puntos, jugadores, valor_total, temporada_actual, fecha_de_carga, user_id[0])
                        cursor.execute(sql_update_user_clasif, data_update)
                        cnn.commit()
                        print(f"Registro de {usuario} actualizado.")
                    else:
                        # Los datos no han cambiado, no hacer nada
                        print(f"Registro de {usuario} ya existe y no ha cambiado.")
                else:
                    # El usuario no tiene un registro en Clasificacion_general, insertar uno nuevo
                    sql_insert_user_clasif = ("INSERT INTO Clasificacion_general "
                                              "(UsuarioID, Posicion, Puntos, Jugadores, ValorTotal, Temporada, f_carga) "
                                              "VALUES (%s, %s, %s, %s, %s, %s, %s)")
                    data_insert = (user_id[0], posicion, puntos, jugadores, valor_total, temporada_actual, fecha_de_carga)
                    cursor.execute(sql_insert_user_clasif, data_insert)
                    cnn.commit()
                    print(f"Registro de {usuario} insertado correctamente.")
            else:
                print(f"El usuario {usuario} no existe en la tabla Usuarios.")
            
    except mysql.connector.Error as error:
        print("Error al insertar datos en la tabla Clasificacion_general:", error)