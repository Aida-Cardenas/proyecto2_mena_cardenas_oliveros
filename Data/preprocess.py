# Data/preprocess.py (fragmento corregido)

import pandas as pd
import difflib
import os

def refine_genre_with_popular(df, df_pop_lookup, cutoff=0.8):

    #1. refinar generos faltantes o genericos con tabla lookup. devuelve df con columna genre depurada. donde;
    # df: dataframe - ventas y columnas de identificacion del juego
    # df_pop_lookup: dataframe lookup  
    # cutoff:  similitud (0..1) para coincidencias difflib

    # 1.1. diccionario lookup (name_clean|year_lookup -> genre_primary)
    df_pop = df_pop_lookup.copy()
    df_pop["key_pop"] = df_pop["name_clean"] + "|" + df_pop["year_lookup"].astype(str)
    pop_dict = dict(zip(df_pop["key_pop"], df_pop["genre_primary"]))
    pop_keys = list(pop_dict.keys())

    # 1.2. definir funcion busqueda para genero primario
    def lookup_popular_genre(name_clean, year_clean):
        if pd.isna(year_clean):
            return None
        target = f"{name_clean}|{int(year_clean)}"
        matches = difflib.get_close_matches(target, pop_keys, n=1, cutoff=cutoff)
        if matches:
            return pop_dict[matches[0]]
        return None

    # 1.3. seleccionar filas con  genero generico (xd) o faltante para refinar
    mask = (df["Genre"].isna()) | (df["Genre"].str.lower() == "misc")
    df_misc = df[mask].copy()

    # 1.4. aplicar lookup  fila por fila para generar nuevos generos
    refined = []
    for idx, row in df_misc.iterrows():
        gen_new = lookup_popular_genre(row["name_clean"], row["year_clean"])
        if gen_new:
            refined.append(gen_new)
        else:
            # si no hay coincidencia conservar genero original
            refined.append(row["Genre"])
    df.loc[mask, "Genre"] = refined
    return df

def merge_3m_8y(df_3m, df_8y):
    # 2. integrar ventas de corto (3m) y largo plazo (8y)
    merged = df_3m.merge(
        df_8y,
        how="outer",
        on=["name_clean", "platform_clean", "year_clean"],
        suffixes=("_3m", "_8y")
    )

    # 2.1. crear columna global_sales
    merged["Global_Sales"] = merged["Global_Sales_8y"].fillna(merged["Global_Sales_3m"])

    # 2.2. crear genre y publisher 
    merged["Genre"] = merged["Genre_8y"].fillna(merged["Genre_3m"])
    merged["Publisher"] = merged["Publisher_8y"].fillna(merged["Publisher_3m"])

    # 2.3. seleccionar columnas para analisis y modelado (jesus christ)
    df_final = merged[
        ["name_clean", "platform_clean", "year_clean", "Genre", "Publisher",
         "Global_Sales", "Critic_Score", "Critic_Count", "User_Score", "User_Count"]
    ].copy()
    return df_final

def save_processed(df, path):
    # 3. guardar dataframe procesado en archivo CSV
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    df.to_csv(path, index=False)

def preprocess_all(df_3m, df_8y, df_pop_lookup):
    # preprocesamiento completo
    # merge
    df_int = merge_3m_8y(df_3m, df_8y)

    # generos
    df_int = refine_genre_with_popular(df_int, df_pop_lookup, cutoff=0.8)

    print("Géneros restantes tras refinar (top 10):")
    print(df_int["Genre"].value_counts().head(10))

    # guardar csv
    output_path = "Processed/vgsales_integrated_refined.csv"
    save_processed(df_int, output_path)
    print(f"Paso 4: Dataset integrado refinado guardado en {output_path}")

    return df_int

if __name__ == "__main__":
    from load_data import load_and_clean
    df3, df8, dfpop = load_and_clean()
    df_final = preprocess_all(df3, df8, dfpop)
