import requests
import pandas as pd
from io import StringIO
from utils import log
from bs4 import BeautifulSoup

BASE_URL = "https://www.football-data.co.uk"
SPAIN_URL = f"{BASE_URL}/spainm.php"


def descargar_urls_csv():
    """Extrae todas las URLs de CSV de LaLiga en football-data.co.uk"""
    resp = requests.get(SPAIN_URL)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    urls = []
    for link in soup.find_all("a"):
        href = link.get("href", "")
        if href.endswith(".csv") and ("SP1" in href or "SP.csv" in href):
            urls.append(BASE_URL + "/" + href.lstrip("/"))
    return urls


def parsear_temporada(url: str) -> str:
    """
    Convierte la ruta completa de football-data.co.uk en temporada 'YY/YY'
    Ejemplo: .../2324/SP1.csv -> '23/24'
    """
    partes = url.split("/")
    for p in partes:
        if p.isdigit() and len(p) == 4:  # carpeta tipo 2324, 2223, etc.
            inicio, fin = p[:2], p[2:]
            return f"{inicio}/{fin}"
    return "??/??"


def api_football_data_co_uk_jornadas(driver=None, schema=None):
    datos = []
    urls = descargar_urls_csv()
    log(f"Encontradas {len(urls)} temporadas de CSV")

    for url in urls:
        temporada = parsear_temporada(url)
        log(f"Descargando {url} → temporada {temporada}")

        resp = requests.get(url)
        if resp.status_code != 200:
            log(f"⚠️ Error {resp.status_code} en {url}")
            continue

        try:
            df = pd.read_csv(
                StringIO(resp.text),
                on_bad_lines="skip",
                engine="python"
            )
        except Exception as e:
            log(f"⚠️ Error leyendo {url}: {e}")
            continue

        df.columns = [c.lower().strip() for c in df.columns]

        if not {"hometeam", "awayteam", "fthg", "ftag"}.issubset(df.columns):
            log(f"⚠️ {url} no tiene columnas esperadas")
            continue

        # Ordenar por fecha si está disponible
        if "date" in df.columns:
            try:
                df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
                df = df.sort_values("date").reset_index(drop=True)
            except Exception:
                pass

        equipos = set(df["hometeam"].unique()) | set(df["awayteam"].unique())
        partidos_por_jornada = len(equipos) // 2 if equipos else 10

        jornada_actual, count = 1, 0
        for _, row in df.iterrows():
            if pd.isna(row["hometeam"]) or pd.isna(row["awayteam"]):
                continue  # saltar filas sin equipos

            gl, gv = row["fthg"], row["ftag"]
            resultado = f"{int(gl)}-{int(gv)}" if pd.notna(gl) and pd.notna(gv) else None

            datos.append({
                "jornada": str(jornada_actual),
                "local": str(row["hometeam"]),
                "visitante": str(row["awayteam"]),
                "resultado": resultado,
                "temporada": temporada
            })

            count += 1
            if count >= partidos_por_jornada:
                jornada_actual += 1
                count = 0

    # 🔥 Limpiamos partidos sin local/visitante
    datos_validos = [
        d for d in datos
        if d["local"] and d["visitante"] and d["local"] != "NaT" and d["visitante"] != "NaT"
    ]
    log(f"Partidos válidos después de limpiar: {len(datos_validos)} (de {len(datos)})")
    return datos_validos



def api_football_data_co_uk_datos_laliga(driver=None, schema=None):
    """
    Calcula la clasificación final por temporada a partir de los partidos
    procesados en api_football_data_co_uk_jornadas
    """
    partidos = api_football_data_co_uk_jornadas()
    df = pd.DataFrame(partidos)

    resultados = []
    for temporada, df_temp in df.groupby("temporada"):
        equipos = set(df_temp["local"].unique()) | set(df_temp["visitante"].unique())
        tabla = {eq: {"pj": 0, "pg": 0, "pe": 0, "pp": 0, "gf": 0, "gc": 0, "pts": 0} for eq in equipos}

        for _, row in df_temp.iterrows():
            local, visitante = row["local"], row["visitante"]
            if row["resultado"] is None:
                continue
            gl, gv = map(int, row["resultado"].split("-"))

            # Actualizamos estadísticas
            tabla[local]["pj"] += 1
            tabla[visitante]["pj"] += 1
            tabla[local]["gf"] += gl
            tabla[local]["gc"] += gv
            tabla[visitante]["gf"] += gv
            tabla[visitante]["gc"] += gl

            if gl > gv:
                tabla[local]["pg"] += 1
                tabla[visitante]["pp"] += 1
                tabla[local]["pts"] += 3
            elif gl < gv:
                tabla[visitante]["pg"] += 1
                tabla[local]["pp"] += 1
                tabla[visitante]["pts"] += 3
            else:
                tabla[local]["pe"] += 1
                tabla[visitante]["pe"] += 1
                tabla[local]["pts"] += 1
                tabla[visitante]["pts"] += 1

        # Convertimos a lista ordenada por puntos, DG y victorias
        df_clasif = pd.DataFrame([
            {
                "equipo": eq,
                "pj": stats["pj"],
                "pg": stats["pg"],
                "pe": stats["pe"],
                "pp": stats["pp"],
                "dg": stats["gf"] - stats["gc"],
                "pts": stats["pts"],
                "temporada": temporada
            }
            for eq, stats in tabla.items()
        ])

        df_clasif = df_clasif.sort_values(
            ["pts", "dg", "pg"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

        df_clasif["posicion"] = df_clasif.index + 1
        df_clasif["escudo"] = None  # no disponible en football-data.co.uk

        resultados.extend(df_clasif.to_dict(orient="records"))

    log(f"Clasificaciones generadas: {len(resultados)}")
    return resultados


def api_football_data_co_uk_jornadas_raw(driver=None, schema=None):
    """
    Descarga todos los CSV de football-data.co.uk y devuelve
    TODOS LOS CAMPOS, listos para dbo.jornadas_liga_raw
    """
    datos = []
    urls = descargar_urls_csv()
    log(f"Encontradas {len(urls)} temporadas de CSV (raw)")

    for url in urls:
        temporada = parsear_temporada(url)
        log(f"Descargando {url} → temporada {temporada}")

        try:
            df = pd.read_csv(
                StringIO(requests.get(url).text),
                on_bad_lines="skip",
                engine="python"
            )
        except Exception as e:
            log(f"⚠️ Error leyendo {url}: {e}")
            continue

        # Normalizamos columnas: lowercase + underscores
        df.columns = [c.lower().strip().replace(" ", "_").replace(".", "_") for c in df.columns]

        # Añadimos temporada
        df["temporada"] = temporada

        # Convertimos a lista de diccionarios
        datos.extend(df.to_dict(orient="records"))

    log(f"Total filas procesadas (raw): {len(datos)}")
    return datos


def api_football_data_co_uk_datos_laliga_raw(driver=None, schema=None):
    """
    Calcula la clasificación final a partir de los datos raw,
    con GF y GC además de PJ, PG, PE, PP, DG y PTS
    """
    partidos = api_football_data_co_uk_jornadas_raw()
    df = pd.DataFrame(partidos)

    # Validar que columnas claves existen
    if not {"hometeam", "awayteam", "fthg", "ftag"}.issubset(df.columns):
        log("⚠️ Columnas necesarias no encontradas en CSV")
        return []

    resultados = []
    for temporada, df_temp in df.groupby("temporada"):
        equipos = set(df_temp["hometeam"].unique()) | set(df_temp["awayteam"].unique())
        tabla = {eq: {"pj": 0, "pg": 0, "pe": 0, "pp": 0, "gf": 0, "gc": 0, "pts": 0} for eq in equipos}

        for _, row in df_temp.iterrows():
            local, visitante = row["hometeam"], row["awayteam"]
            if pd.isna(local) or pd.isna(visitante):
                continue
            if pd.isna(row["fthg"]) or pd.isna(row["ftag"]):
                continue

            gl, gv = int(row["fthg"]), int(row["ftag"])

            tabla[local]["pj"] += 1
            tabla[visitante]["pj"] += 1
            tabla[local]["gf"] += gl
            tabla[local]["gc"] += gv
            tabla[visitante]["gf"] += gv
            tabla[visitante]["gc"] += gl

            if gl > gv:
                tabla[local]["pg"] += 1
                tabla[visitante]["pp"] += 1
                tabla[local]["pts"] += 3
            elif gl < gv:
                tabla[visitante]["pg"] += 1
                tabla[local]["pp"] += 1
                tabla[visitante]["pts"] += 3
            else:
                tabla[local]["pe"] += 1
                tabla[visitante]["pe"] += 1
                tabla[local]["pts"] += 1
                tabla[visitante]["pts"] += 1

        df_clasif = pd.DataFrame([
            {
                "equipo": eq,
                "pj": stats["pj"],
                "pg": stats["pg"],
                "pe": stats["pe"],
                "pp": stats["pp"],
                "gf": stats["gf"],
                "gc": stats["gc"],
                "dg": stats["gf"] - stats["gc"],
                "pts": stats["pts"],
                "temporada": temporada
            }
            for eq, stats in tabla.items()
        ])

        df_clasif = df_clasif.sort_values(
            ["pts", "dg", "pg"], ascending=[False, False, False]
        ).reset_index(drop=True)
        df_clasif["posicion"] = df_clasif.index + 1
        df_clasif["escudo"] = None

        resultados.extend(df_clasif.to_dict(orient="records"))

    log(f"Clasificaciones generadas (raw): {len(resultados)}")
    return resultados