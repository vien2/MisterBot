import pandas as pd
from utils import conexion_db
from psycopg2.extras import execute_values

def cargar_dataframe_postgresql(df, schema, tabla, clave_conflicto=None):
    if df.empty:
        print(f"⚠️ DataFrame vacío. No se insertaron registros en {schema}.{tabla}.")
        return

    with conexion_db() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT column_name FROM information_schema.columns 
                WHERE table_schema = %s AND table_name = %s
            """, (schema, tabla))
            columnas_validas = [row[0] for row in cur.fetchall()]

            columnas = [col for col in df.columns if col in columnas_validas]
            df = df[columnas]

            columnas_sql = ', '.join([f'"{col}"' for col in columnas])
            df = df.where(pd.notnull(df), None)
            registros = [tuple(x) for x in df.to_numpy()]

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
            conn.commit()
            print(f"💾 {len(registros)} registros insertados en {schema}.{tabla}")