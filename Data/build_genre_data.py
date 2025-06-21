# src/data/build_popular_lookup.py

import pandas as pd
import json
import difflib

def clean_text(s: str) -> str:
    return str(s).strip().lower()

def parse_primary_genre(genres_str: str) -> str:
    """
    Toma una cadena como "['Adventure','RPG']" y devuelve 'Adventure'.
    Si no es un formato JSON válido, da None.
    """
    try:
        # Reemplazamos comillas simples por dobles para que json.loads funcione
        arr = json.loads(genres_str.replace("'", '"'))
        if isinstance(arr, list) and arr:
            return arr[0]
    except:
        pass
    return None

def build_popular_lookup(popular_path: str) -> pd.DataFrame:
    """
    Carga popularvg2y.csv y devuelve un DataFrame con columnas:
      ['name_clean', 'year_lookup', 'genre_primary']
    """
    df_pop = pd.read_csv(popular_path)
    # 1. Limpiar Title
    df_pop["name_clean"] = df_pop["Title"].map(clean_text)
    # 2. Extraer año de Release Date
    df_pop["year_lookup"] = pd.to_datetime(
        df_pop["Release Date"], errors="coerce"
    ).dt.year.astype("Int64")
    # 3. Extraer primer género
    df_pop["genre_primary"] = df_pop["Genres"].map(parse_primary_genre)
    # 4. Seleccionar solo filas con año válido y género primario no nulo
    df_pop = df_pop.dropna(subset=["year_lookup", "genre_primary"])
    # 5. Eliminar duplicados: si el mismo (name_clean, year_lookup) aparece varias veces,
    #    dejamos el primero (o podrías escoger el más repetido, pero aquí basta el primero).
    df_lookup = df_pop[["name_clean", "year_lookup", "genre_primary"]].drop_duplicates(
        subset=["name_clean", "year_lookup"], keep="first"
    )
    return df_lookup

if __name__ == "__main__":
    lookup = build_popular_lookup("data/raw/popularvg2y.csv")
    print("Primeras filas de popular-lookup:")
    print(lookup.head())
    lookup.to_csv("data/interim/popular_genre_lookup.csv", index=False)
