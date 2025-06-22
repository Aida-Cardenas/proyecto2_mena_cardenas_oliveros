# Features/build_features.py

import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_and_save_features():
    # leer dataset ventas 
    df = pd.read_csv("./Data/Processed/vgsales_integrated_refined.csv")

    # is_topseller para cada juego, asignar 1 si sus ventas superan el percentil 75 de su genero
    percentiles_75 = (
        df.groupby("Genre")["Global_Sales"]
          .quantile(0.75)
          .to_dict()
    )

    def assign_topseller(row):
        umbral = percentiles_75.get(row["Genre"], 0)
        return 1 if row["Global_Sales"] >= umbral else 0

    df["is_topseller"] = df.apply(assign_topseller, axis=1)

    # input y limpiar datos numericos
    numeric_raw = [
        "Critic_Score", "Critic_Count",
        "User_Score", "User_Count",
        "Global_Sales"
    ]
    for col in numeric_raw:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in numeric_raw:
        if col in df.columns:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    # caracteristicas de ventas historicas
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

    # variable de interaccion genero x plataforma
    df["genre_platform_interaction"] = df["Genre"] + "_" + df["platform_clean"]

    # codificar variables categoricas con one-hot encoding
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

    # estandarizar variables numericas con StandardScaler
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

    # preparar conjuntos de entrenamiento
    X = pd.concat([df_ohe, df_scaled], axis=1)
    y = df["is_topseller"]

    # guardar matrices en csv
    os.makedirs("Data/Processed", exist_ok=True)
    X.to_csv("Data/Processed/features_matrix.csv", index=False)
    y.to_csv("Data/Processed/labels.csv", index=False)

    print("Feature engineering completado:")
    print("- Features guardadas en Data/Processed/features_matrix.csv")
    print("- Labels   guardadas en Data/Processed/labels.csv")

if __name__ == "__main__":
    build_and_save_features()
