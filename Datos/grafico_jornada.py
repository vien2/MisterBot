import mysql.connector
import matplotlib.pyplot as plt


def grafico_jornada(cnn):
  # Definir la consulta SQL para la evolución de puntos por jornada
  sql_query_evolucion = """
      SELECT b.Nombre as Nombre, a.jornada, a.puntos
      FROM clasificacion_jornada a
      INNER JOIN usuarios b ON a.UsuarioID = b.id
      ORDER BY a.jornada ASC
  """

  # Definir la consulta SQL para los puntos máximos de cada usuario
  sql_query_puntos_maximos = """
      SELECT UsuarioID, MAX(puntos) as max_puntos
      FROM clasificacion_general
      GROUP BY UsuarioID
  """

  # Crear un cursor para ejecutar las consultas
  cursor = cnn.cursor()

  # Ejecutar la consulta de evolución de puntos por jornada
  cursor.execute(sql_query_evolucion)
  results_evolucion = cursor.fetchall()

  # Ejecutar la consulta de puntos máximos por usuario
  cursor.execute(sql_query_puntos_maximos)
  results_puntos_maximos = cursor.fetchall()

  # Cerrar la conexión a la base de datos
  cnn.close()

  # Crear un diccionario para almacenar los puntos por usuario
  puntos_por_usuario = {}

  # Procesar los resultados de evolución de puntos por jornada
  for row in results_evolucion:
      nombre, jornada, puntos = row
      if nombre not in puntos_por_usuario:
          puntos_por_usuario[nombre] = {"posicion": None, "puntos": []}
      puntos_por_usuario[nombre]["puntos"].append(puntos)

  # Procesar los resultados de puntos máximos por usuario
  puntos_maximos = {str(row[0]): row[1] for row in results_puntos_maximos}

  # Asignar la posición en la clasificación general a cada usuario
  for nombre, data in puntos_por_usuario.items():
      max_puntos = puntos_maximos.get(nombre, None)
      data["posicion"] = max_puntos

  # Crear gráficos para la evolución de puntos por usuario
  for nombre, data in puntos_por_usuario.items():
      if data["posicion"] is not None:
          plt.plot(data["posicion"], data["puntos"], label=nombre)

  # Configurar la leyenda y etiquetas
  plt.legend(loc="best")
  plt.xlabel("Posición en Clasificación General")
  plt.ylabel("Puntos")
  plt.title("Evolución de Puntos por Posición en Clasificación General")
  plt.grid(True)

  # Mostrar el gráfico
  plt.show()
