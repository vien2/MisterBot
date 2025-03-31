import os
import pandas as pd
import shutil
from utils import conexion_db, log
from psycopg2.extras import execute_values

def cargar_urls_jugadores_csv(ruta_csv, tipo_carga):
    tabla = "urls_jugadores"
    schema = "dbo"
    incremental_field = "id_jugador"
    hash_field = "hash"

    try:
        df = pd.read_csv(ruta_csv)
        df.columns = [col.lower() for col in df.columns]

        if df.empty:
            log(f"No se cargaron datos porque el archivo {ruta_csv} está vacío.")
            return

        with conexion_db() as conn:
            with conn.cursor() as cur:

                if tipo_carga == "total":
                    # 🔹 Línea 20: DELETE para carga total
                    cur.execute(f"DELETE FROM {schema}.{tabla}")
                    log(f"Se eliminó el contenido previo de {schema}.{tabla} para carga total.")

                elif tipo_carga == "incremental":
                    # 🔹 Línea 24–31: Filtrado según valor incremental
                    cur.execute(f"""
                        SELECT valor_incremental 
                        FROM {schema}.incremental_load_info 
                        WHERE tabla = %s
                    """, (tabla,))
                    resultado = cur.fetchone()
                    if resultado:
                        valor_incremental = int(resultado[0])
                        df = df[df[incremental_field] > valor_incremental]
                        log(f"Carga incremental - registros con {incremental_field} > {valor_incremental}")
                    else:
                        log("No se encontró valor incremental anterior, se cargará todo.")

                elif tipo_carga == "diferencial":
                    # 🔹 Línea 35–42: Comparación de hashes con tabla destino
                    cur.execute(f"SELECT id_jugador, temporada, {hash_field} FROM {schema}.{tabla}")
                    registros_db = {(str(r[0]), r[1]): r[2] for r in cur.fetchall()}
                    log(f"Carga diferencial - registros existentes: {len(registros_db)}")

                    df["clave"] = list(zip(df[incremental_field].astype(str), df["temporada"]))
                    df = df[df["clave"].apply(lambda x: x not in registros_db or df.loc[df["clave"] == x, hash_field].values[0] != registros_db.get(x))]
                    df.drop(columns=["clave"], inplace=True)

                    log(f"Carga diferencial - registros nuevos o modificados: {len(df)}")

                if df.empty:
                    log("No hay nuevos registros para cargar tras aplicar lógica de carga.")
                    return

                # 🔹 Línea 48–55: Insert para total / incremental / diferencial
                columnas = list(df.columns)
                registros = [tuple(x) for x in df.to_numpy()]
                columnas_sql = ', '.join([f'"{col}"' for col in columnas])
                insert_query = f"""
                    INSERT INTO {schema}.{tabla} ({columnas_sql})
                    VALUES %s
                    ON CONFLICT (id_jugador, temporada) DO UPDATE SET 
                        {', '.join([f'"{col}" = EXCLUDED."{col}"' for col in columnas if col not in ("id_jugador", "temporada")])};
                """
                execute_values(cur, insert_query, registros)

                if tipo_carga == "incremental":
                    # 🔹 Línea 59–64: Actualizar incremental_load_info
                    max_id = int(df[incremental_field].max())
                    cur.execute(f"""
                        INSERT INTO {schema}.incremental_load_info(tabla, valor_incremental)
                        VALUES (%s, %s)
                        ON CONFLICT (tabla) DO UPDATE SET valor_incremental = EXCLUDED.valor_incremental;
                    """, (tabla, max_id))

                conn.commit()

        # 🔹 Línea 68–71: Mover archivo si fue exitoso
        destino_ok = ruta_csv.replace("data/", "data/ok/")
        os.makedirs(os.path.dirname(destino_ok), exist_ok=True)
        shutil.move(ruta_csv, destino_ok)
        log(f"Carga exitosa. Archivo movido a: {destino_ok}")

    except Exception as e:
        # 🔹 Línea 74–78: Mover archivo a /ko si falla
        destino_ko = ruta_csv.replace("data/", "data/ko/")
        os.makedirs(os.path.dirname(destino_ko), exist_ok=True)
        shutil.move(ruta_csv, destino_ko)
        log(f"Error durante la carga: {e}")
        log(f"Archivo movido a: {destino_ko}")
