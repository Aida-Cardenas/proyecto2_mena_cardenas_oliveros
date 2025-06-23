# src/data/build_popular_lookup.py

import pandas as pd
import json
import difflib

def clean_text(s: str) -> str:
    return str(s).strip().lower()

def parse_primary_genre(genres_str: str) -> str:
    # se toma cadena "['Adventure','RPG']" y devuelve 'Adventure'
    try:
        arr = json.loads(genres_str.replace("'", '"'))
        if isinstance(arr, list) and arr:
            return arr[0]
    except:
        pass
    return None

def build_popular_lookup(popular_path: str) -> pd.DataFrame:
    # carga el archivo popularvg2y.csv -> dataframe columnas: ['name_clean', 'year_lookup', 'genre_primary']
    df_pop = pd.read_csv(popular_path)
    # 1. limpiar titulo
    df_pop["name_clean"] = df_pop["Title"].map(clean_text)
    # 2. extraer ano release
    df_pop["year_lookup"] = pd.to_datetime(
        df_pop["Release Date"], errors="coerce"
    ).dt.year.astype("Int64")
    # 3. extraer primer genero/principal
    df_pop["genre_primary"] = df_pop["Genres"].map(parse_primary_genre)
    # 4.validacion
    df_pop = df_pop.dropna(subset=["year_lookup", "genre_primary"])
    # 5. eliminar duplicates
    df_lookup = df_pop[["name_clean", "year_lookup", "genre_primary"]].drop_duplicates(
        subset=["name_clean", "year_lookup"], keep="first"
    )
    return df_lookup

if __name__ == "__main__":
    lookup = build_popular_lookup("data/raw/popularvg2y.csv")
    print("Primeras filas de popular-lookup:")
    print(lookup.head())
    lookup.to_csv("data/interim/popular_genre_lookup.csv", index=False)
