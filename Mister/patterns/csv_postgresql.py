import pandas as pd
import os, shutil
from utils import conexion_db, log, limpiar_columna
from psycopg2.extras import execute_values

def cargar_csv_postgresql(ruta_csv, schema, tabla, tipo_carga, incremental_field=None, clave_conflicto=None, hash_field=None):
    try:
        df = pd.read_csv(ruta_csv)
        log(f"Preparando carga para tabla {schema}.{tabla} con {len(df)} registros")
        df.columns = [limpiar_columna(col) for col in df.columns]

        if df.empty:
            log(f"No se cargaron datos porque el archivo {ruta_csv} está vacío.")
            return

        with conexion_db() as conn:
            with conn.cursor() as cur:

                if tipo_carga == "total":
                    cur.execute(f"DELETE FROM {schema}.{tabla}")
                    log(f"Carga total: datos previos eliminados de {schema}.{tabla}")

                elif tipo_carga == "incremental" and incremental_field:
                    cur.execute(f"""
                        SELECT valor_incremental FROM {schema}.incremental_load_info 
                        WHERE tabla = %s
                    """, (tabla,))
                    resultado = cur.fetchone()
                    if resultado:
                        valor_incremental = resultado[0]

                        if isinstance(valor_incremental, str):
                            try:
                                valor_incremental_dt = pd.to_datetime(valor_incremental, errors='coerce')
                                if not pd.isnull(valor_incremental_dt):
                                    df[incremental_field] = pd.to_datetime(df[incremental_field], errors='coerce')
                                    valor_incremental = valor_incremental_dt
                            except Exception:
                                pass

                        log(f"Carga incremental: filtrando {incremental_field} > {valor_incremental}")
                        df = df[df[incremental_field] > valor_incremental]
                    else:
                        log("Carga incremental: sin valor previo, se cargará todo")

                elif tipo_carga == "diferencial" and hash_field and clave_conflicto:
                    conflicto_cols = ', '.join(clave_conflicto)
                    cur.execute(f"SELECT {conflicto_cols}, {hash_field} FROM {schema}.{tabla}")
                    registros_db = {
                        tuple(map(str, row[:-1])): row[-1] for row in cur.fetchall()
                    }
                    log(f"Carga diferencial: registros en DB: {len(registros_db)}")

                    df["__clave__"] = df[clave_conflicto].astype(str).apply(tuple, axis=1)
                    df = df[df.apply(
                        lambda row: registros_db.get(tuple(row[clave_conflicto].astype(str))) != row[hash_field],
                        axis=1
                    )]
                    df.drop(columns=["__clave__"], inplace=True)
                    log(f"Carga diferencial: nuevos o modificados: {len(df)}")

                elif tipo_carga == "acumulado":
                    log("Carga acumulada: se insertarán todos los registros tal cual vienen.")

                if df.empty:
                    log("Carga: No hay registros nuevos que cargar tras filtros.")
                    destino_ok = ruta_csv.replace("data/", "data/ok/")
                    os.makedirs(os.path.dirname(destino_ok), exist_ok=True)
                    shutil.move(ruta_csv, destino_ok)
                    log(f"Carga exitosa. Archivo movido a: {destino_ok}")
                    return

                cur.execute(f"""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_schema = %s AND table_name = %s
                """, (schema, tabla))
                columnas_validas = [row[0] for row in cur.fetchall()]

                df = df[[col for col in columnas_validas if col in df.columns]]

                columnas = list(df.columns)
                columnas_sql = ', '.join([f'"{col}"' for col in columnas])
                df = df.where(pd.notnull(df), None)
                registros = [tuple(x) for x in df.to_numpy()]

                conflict_clause = ""
                if tipo_carga in ("total", "incremental", "diferencial","acumulado") and clave_conflicto:
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

        destino_ok = ruta_csv.replace("data/", "data/ok/")
        os.makedirs(os.path.dirname(destino_ok), exist_ok=True)
        shutil.move(ruta_csv, destino_ok)
        log(f"Carga exitosa. Archivo movido a: {destino_ok}")
        return True

    except Exception as e:
        destino_ko = ruta_csv.replace("data/", "data/ko/")
        os.makedirs(os.path.dirname(destino_ko), exist_ok=True)
        shutil.move(ruta_csv, destino_ko)
        log(f"Error durante la carga: {e}")
        log(f"Archivo movido a: {destino_ko}")
        return False
