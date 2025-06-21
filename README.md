# Clasificador de Videojuegos

Este proyecto implementa un sistema de clasificación de videojuegos utilizando técnicas de aprendizaje automático. El sistema está diseñado para analizar y clasificar juegos basándose en diferentes características y géneros.
```

## Descripción de Componentes

### Datos
- **Raw**: Contiene los conjuntos de datos originales sin procesar
  - `vgsales3m.csv`: Datos de ventas de videojuegos
  - `vgsales8years.csv`: Histórico de ventas de 8 años
  - `popularvg2y.csv`: Datos de popularidad de videojuegos

- **Interim**: Datos en proceso de transformación
  - `popular_genre_lookup.csv`: Tabla de búsqueda de géneros populares

- **Processed**: Datos procesados listos para el entrenamiento
  - `features_matrix.csv`: Matriz de características
  - `labels.csv`: Etiquetas para clasificación
  - `vgsales_integrated_clean.csv`: Datos de ventas limpios
  - `vgsales_integrated_refined.csv`: Datos de ventas refinados

### Características
- `build_features.py`: Script para la extracción y construcción de características

### Modelos
- Implementación de dos modelos de clasificación:
  - Perceptrón básico (`baseline_perceptron.pth`)
  - Clasificador MLP (`mlp_classifier.pth`)
- Script de entrenamiento utilizando PyTorch

## Requisitos

- Python 3.x
- PyTorch
- Pandas
- NumPy
- Scikit-learn

## Uso

1. Preprocesamiento de datos:
```bash
python Data/preprocess.py
```

2. Construcción de características:
```bash
python Features/build_features.py
```

3. Entrenamiento del modelo:
```bash
python Models/train_pytorch.py
```
