# Features/build_features.py

import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_and_save_features():
    # Paso 1: Leer dataset refinado de ventas consolidadas
    #         CSV con datos ya integrados y géneros refinados
    df = pd.read_csv("./Data/Processed/vgsales_integrated_refined.csv")

    # Paso 2: Definir etiqueta 'is_topseller'
    #         Para cada juego, asignar 1 si sus ventas superan el percentil 75 de su género
    percentiles_75 = (
        df.groupby("Genre")["Global_Sales"]
          .quantile(0.75)
          .to_dict()
    )

    #    Función auxiliar para asignar etiqueta
    def assign_topseller(row):
        umbral = percentiles_75.get(row["Genre"], 0)
        return 1 if row["Global_Sales"] >= umbral else 0

    df["is_topseller"] = df.apply(assign_topseller, axis=1)

    # Paso 3: Imputar y limpiar datos numéricos originales
    #         Convertir scores y ventas a numérico, forzar errores a NaN y luego imputar medianas
    numeric_raw = [
        "Critic_Score", "Critic_Count",
        "User_Score", "User_Count",
        "Global_Sales"
    ]
    for col in numeric_raw:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    #    Rellenar NaN con la mediana de cada columna numérica
    for col in numeric_raw:
        if col in df.columns:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    # Paso 4: Crear características de ventas históricas
    #         Media de ventas globales agrupada por género y por plataforma ('platform_clean')
    ventas_por_genero = (
        df.groupby("Genre")["Global_Sales"]
          .mean()
          .rename("mean_sales_by_genre")
    )
    ventas_por_plataforma = (
        df.groupby("platform_clean")["Global_Sales"]
          .mean()
          .rename("mean_sales_by_platform")
    )
    df = df.merge(ventas_por_genero, on="Genre", how="left")
    df = df.merge(ventas_por_plataforma, on="platform_clean", how="left")

    # Paso 5: Construir variable de interacción Género×Plataforma
    #         Combinar categorías para capturar relaciones conjuntas entre género y plataforma
    df["genre_platform_interaction"] = df["Genre"] + "_" + df["platform_clean"]

    # Paso 6: Codificar variables categóricas con One-Hot Encoding
    #         Cada valor único se convierte en columna binaria para modelos ML
    categorical_cols = [
        "Genre", "platform_clean", "Publisher", "genre_platform_interaction"
    ]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe_matrix = ohe.fit_transform(df[categorical_cols])
    ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
    df_ohe = pd.DataFrame(
        ohe_matrix,
        columns=ohe_feature_names,
        index=df.index
    )

    # Paso 7: Estándarizar variables numéricas con StandardScaler
    #         (valor - media) / desviación estándar para normalizar rangos
    numeric_cols = [
        "Critic_Score", "Critic_Count",
        "User_Score", "User_Count",
        "mean_sales_by_genre", "mean_sales_by_platform",
        "Global_Sales"
    ]
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols]),
        columns=numeric_cols,
        index=df.index
    )

    # Paso 8: Preparar conjuntos de entrenamiento
    #         X = matriz de características (OHE + escaladas), y = vector binario de etiquetas
    X = pd.concat([df_ohe, df_scaled], axis=1)
    y = df["is_topseller"]

    # Paso 9: Guardar matrices en CSV para entrenamiento
    #         'features_matrix.csv' y 'labels.csv' en carpeta data/processed
    os.makedirs("data/processed", exist_ok=True)
    X.to_csv("data/processed/features_matrix.csv", index=False)
    y.to_csv("data/processed/labels.csv", index=False)

    print("Feature engineering completado:")
    print("- Features guardadas en data/processed/features_matrix.csv")
    print("- Labels   guardadas en data/processed/labels.csv")

if __name__ == "__main__":
    build_and_save_features()
