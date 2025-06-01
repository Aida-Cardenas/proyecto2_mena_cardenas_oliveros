# Data/preprocess.py (fragmento corregido)

import pandas as pd
import difflib
import os

def refine_genre_with_popular(df, df_pop_lookup, cutoff=0.8):
    """
    Paso 1: Refinar géneros faltantes o genéricos ('Misc') usando tabla de lookup.
    - df: DataFrame integrado con ventas y columnas clave ['Genre','name_clean','year_clean'].
    - df_pop_lookup: DataFrame de lookup con ['name_clean','year_lookup','genre_primary'].
    - cutoff: umbral de similitud (0..1) para coincidencia difusa (difflib).
    Devuelve df con columna 'Genre' refinada.
    """
    # Paso 1.1: Construir diccionario de lookup ('name_clean|year_lookup' → 'genre_primary')
    df_pop = df_pop_lookup.copy()
    df_pop["key_pop"] = df_pop["name_clean"] + "|" + df_pop["year_lookup"].astype(str)
    pop_dict = dict(zip(df_pop["key_pop"], df_pop["genre_primary"]))
    pop_keys = list(pop_dict.keys())

    # Paso 1.2: Definir función de búsqueda difusa en keys para asignar género primario
    def lookup_popular_genre(name_clean, year_clean):
        # Si año de lanzamiento es NaN, no hay referencia para lookup
        if pd.isna(year_clean):
            return None
        target = f"{name_clean}|{int(year_clean)}"
        matches = difflib.get_close_matches(target, pop_keys, n=1, cutoff=cutoff)
        if matches:
            return pop_dict[matches[0]]
        return None

    # Paso 1.3: Seleccionar filas con 'Genre' genérico o faltante para refinar
    mask = (df["Genre"].isna()) | (df["Genre"].str.lower() == "misc")
    df_misc = df[mask].copy()

    # Paso 1.4: Aplicar lookup difuso fila por fila para generar nuevos géneros
    refined = []
    for idx, row in df_misc.iterrows():
        gen_new = lookup_popular_genre(row["name_clean"], row["year_clean"])
        if gen_new:
            refined.append(gen_new)
        else:
            # No se encontró coincidencia: conservar género original (None o 'Misc')
            refined.append(row["Genre"])
    # Paso 1.5: Reemplazar valores en df original para las filas seleccionadas
    df.loc[mask, "Genre"] = refined
    return df

def merge_3m_8y(df_3m, df_8y):
    """
    Paso 2: Integrar ventas de corto (3m) y largo plazo (8y)
    - Outer join para conservar todas las observaciones de ambos data sources.
    - Consolidar 'Global_Sales': preferir df_8y, si no está, usar df_3m.
    - Consolidar 'Genre' y 'Publisher' priorizando datos más recientes (8y).
    Devuelve df integrado.
    """
    merged = df_3m.merge(
        df_8y,
        how="outer",
        on=["name_clean", "platform_clean", "year_clean"],
        suffixes=("_3m", "_8y")
    )

    # Paso 2.1: Consolidar columna 'Global_Sales' eligiendo valores de 8y o caídas a 3m
    merged["Global_Sales"] = merged["Global_Sales_8y"].fillna(merged["Global_Sales_3m"])

    # Paso 2.2: Consolidar 'Genre' y 'Publisher' priorizando 8y, luego 3m
    merged["Genre"] = merged["Genre_8y"].fillna(merged["Genre_3m"])
    merged["Publisher"] = merged["Publisher_8y"].fillna(merged["Publisher_3m"])

    # Paso 2.3: Seleccionar columnas finales de interés para análisis/modelado
    df_final = merged[
        ["name_clean", "platform_clean", "year_clean", "Genre", "Publisher",
         "Global_Sales", "Critic_Score", "Critic_Count", "User_Score", "User_Count"]
    ].copy()
    return df_final

def save_processed(df, path):
    """
    Paso 3: Guardar DataFrame procesado en archivo CSV
    - Crea directorio destino si no existe.
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    df.to_csv(path, index=False)

def preprocess_all(df_3m, df_8y, df_pop_lookup):
    """
    Orquestar flujo completo de preprocesamiento:
    1. Integrar datos de ventas (3m + 8y)
    2. Refinar géneros con tabla popular lookup
    3. Guardar dataset refinado para etapas posteriores
    """
    # 1. Merge
    df_int = merge_3m_8y(df_3m, df_8y)

    # 2. Refinar géneros
    df_int = refine_genre_with_popular(df_int, df_pop_lookup, cutoff=0.8)

    # 3. Mostrar recuento de géneros tras refinar
    print("Géneros restantes tras refinar (top 10):")
    print(df_int["Genre"].value_counts().head(10))

    # 4. Guardar CSV integrado refinado
    output_path = "Processed/vgsales_integrated_refined.csv"
    save_processed(df_int, output_path)
    print(f"Paso 4: Dataset integrado refinado guardado en {output_path}")

    return df_int

if __name__ == "__main__":
    # Ejecución standalone: carga, preprocesa y guarda
    from load_data import load_and_clean
    df3, df8, dfpop = load_and_clean()
    df_final = preprocess_all(df3, df8, dfpop)
