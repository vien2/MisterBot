import pandas as pd
import os, shutil
from utils import conexion_db, log
from psycopg2.extras import execute_values

def cargar_csv_postgresql(ruta_csv, schema, tabla, tipo_carga, incremental_field=None, clave_conflicto=None, hash_field=None):
    try:
        df = pd.read_csv(ruta_csv)
        df.columns = [col.lower() for col in df.columns]

        if df.empty:
            log(f"No se cargaron datos porque el archivo {ruta_csv} está vacío.")
            return

        with conexion_db() as conn:
            with conn.cursor() as cur:

                # Eliminar datos anteriores en carga total
                if tipo_carga == "total":
                    cur.execute(f"DELETE FROM {schema}.{tabla}")
                    log(f"Carga total: datos previos eliminados de {schema}.{tabla}")

                # Carga incremental
                elif tipo_carga == "incremental" and incremental_field:
                    cur.execute(f"""
                        SELECT valor_incremental FROM {schema}.incremental_load_info 
                        WHERE tabla = %s
                    """, (tabla,))
                    resultado = cur.fetchone()
                    if resultado:
                        valor_incremental = resultado[0]

                        # Convertimos el tipo de la columna en el DataFrame para que sea comparable
                        if isinstance(valor_incremental, str):
                            try:
                                # Intentamos parsear como datetime si tiene formato de fecha
                                valor_incremental_dt = pd.to_datetime(valor_incremental, errors='coerce')
                                if not pd.isnull(valor_incremental_dt):
                                    df[incremental_field] = pd.to_datetime(df[incremental_field], errors='coerce')
                                    valor_incremental = valor_incremental_dt
                            except Exception:
                                pass  # si no es parseable como fecha, lo tratamos como texto

                        log(f"Carga incremental: filtrando {incremental_field} > {valor_incremental}")
                        df = df[df[incremental_field] > valor_incremental]

                    else:
                        log("Carga incremental: sin valor previo, se cargará todo")

                # Carga diferencial
                elif tipo_carga == "diferencial" and hash_field and clave_conflicto:
                    conflicto_cols = ', '.join(clave_conflicto)
                    cur.execute(f"SELECT {conflicto_cols}, {hash_field} FROM {schema}.{tabla}")
                    registros_db = {
                        tuple(map(str, row[:-1])): row[-1] for row in cur.fetchall()
                    }
                    log(f"Carga diferencial: registros en DB: {len(registros_db)}")

                    df["__clave__"] = df[clave_conflicto].astype(str).agg("-".join, axis=1)
                    df = df[df["__clave__"].apply(
                        lambda k: k not in registros_db or registros_db[k] != df.loc[df["__clave__"] == k, hash_field].values[0]
                    )]
                    df.drop(columns=["__clave__"], inplace=True)
                    log(f"Carga diferencial: nuevos o modificados: {len(df)}")

                if df.empty:
                    log("Carga: No hay registros nuevos que cargar tras filtros.")
                    return

                # 💡 Filtrar columnas que existen en destino
                cur.execute(f"""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_schema = %s AND table_name = %s
                """, (schema, tabla))
                columnas_validas = [row[0] for row in cur.fetchall()]

                # Nos quedamos solo con columnas que existen en la tabla destino
                df = df[[col for col in df.columns if col in columnas_validas]]

                columnas = list(df.columns)
                columnas_sql = ', '.join([f'"{col}"' for col in columnas])
                registros = [tuple(x) for x in df.to_numpy()]

                # Generar cláusula ON CONFLICT
                conflict_clause = ""
                if clave_conflicto:
                    conflict_keys = ', '.join([f'"{col}"' for col in clave_conflicto])
                    updates = ', '.join([f'"{col}" = EXCLUDED."{col}"' for col in columnas if col not in clave_conflicto])
                    conflict_clause = f'ON CONFLICT ({conflict_keys}) DO UPDATE SET {updates}'

                insert_query = f"""
                    INSERT INTO {schema}.{tabla} ({columnas_sql})
                    VALUES %s
                    {conflict_clause};
                """
                execute_values(cur, insert_query, registros)

                if tipo_carga == "incremental" and incremental_field:
                    max_value = df[incremental_field].max()
                    cur.execute(f"""
                        INSERT INTO {schema}.incremental_load_info(tabla, valor_incremental)
                        VALUES (%s, %s)
                        ON CONFLICT (tabla) DO UPDATE SET valor_incremental = EXCLUDED.valor_incremental;
                    """, (tabla, str(max_value)))

                conn.commit()

        # ✅ Mover archivo a OK
        destino_ok = ruta_csv.replace("data/", "data/ok/")
        os.makedirs(os.path.dirname(destino_ok), exist_ok=True)
        shutil.move(ruta_csv, destino_ok)
        log(f"Carga exitosa. Archivo movido a: {destino_ok}")

    except Exception as e:
        destino_ko = ruta_csv.replace("data/", "data/ko/")
        os.makedirs(os.path.dirname(destino_ko), exist_ok=True)
        shutil.move(ruta_csv, destino_ko)
        log(f"Error durante la carga: {e}")
        log(f"Archivo movido a: {destino_ko}")
