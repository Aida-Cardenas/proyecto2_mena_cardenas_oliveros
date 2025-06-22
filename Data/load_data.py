# src/data/load_data.py

import pandas as pd

def clean_text(s: str) -> str:
    return str(s).strip().lower()

def load_and_clean():
    # 1. leer archivos CSV de ventas 
    #         vgsales3m.csv -> 3 meses y vgsales8years.csv -> 8 años de historial
    df_3m = pd.read_csv("data/raw/vgsales3m.csv")
    df_8y = pd.read_csv("data/raw/vgsales8years.csv")
    # 2.cargar lookup de generos populares // tabla creada con build_genre_data.py para refinar valores misc o posibles faltantes
    df_pop_lookup = pd.read_csv("data/interim/popular_genre_lookup.csv")

    # 3. normalizar columnas en df_3m // name_clean y platform_clean -> texto en minusculas y sin espacios extra // year_clean: convertir a entero
    df_3m["name_clean"]     = df_3m["Name"].map(clean_text)
    df_3m["platform_clean"] = df_3m["Platform"].map(clean_text)
    df_3m["year_clean"]     = pd.to_numeric(df_3m["Year"], errors="coerce").astype("Int64")

    # 4.normalizar columnas en df_8y 
    df_8y["name_clean"]     = df_8y["Name"].map(clean_text)
    df_8y["platform_clean"] = df_8y["Platform"].map(clean_text)
    df_8y["year_clean"]     = pd.to_numeric(df_8y["Year_of_Release"], errors="coerce").astype("Int64")

    return df_3m, df_8y, df_pop_lookup
