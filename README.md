# 🎮 Sistema Predictor de Éxito en Videojuegos

## Descripción General

Este sistema utiliza redes neuronales multicapa (MLP) para predecir si un videojuego será un "top-seller" dentro de su categoría de género, basándose en características como género, plataforma, puntuaciones de críticos y usuarios, y datos históricos de ventas.

### Características Principales

- **Modelo MLP**: Red neuronal de 2 capas ocultas con activación ReLU y dropout
- **Modelo Baseline**: Perceptrón simple para comparación
- **Indicadores Predictivos**: Análisis detallado por género y plataforma
- **Interfaz Gráfica**: GUI intuitiva para predicciones individuales
- **Sistema de Validación**: Suite completa de pruebas automáticas
- **Reportes**: Documentación HTML y visualizaciones

## 📋 Requisitos del Sistema

### Dependencias de Python
```bash
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

### Instalación
```bash
# Clonar el repositorio
git clone <repository-url>
cd proyectojejeje

# Instalar dependencias
pip install -r requirements.txt

# O instalar manualmente
pip install torch pandas numpy scikit-learn matplotlib seaborn
```

## 🚀 Guía de Uso Rápido

### Configuración Inicial Completa
```bash
python run_system.py --full
```

Este comando ejecuta todo el pipeline:
1. Preparación y procesamiento de datos
2. Entrenamiento de modelos (MLP + Baseline)
3. Generación de indicadores predictivos
4. Validación del sistema
5. Lanzamiento de la interfaz gráfica

### Uso de Componentes Individuales

```bash
# Solo preparar datos
python run_system.py --prepare

# Solo entrenar modelos
python run_system.py --train

# Solo generar indicadores
python run_system.py --indicators

# Solo ejecutar validación
python run_system.py --validate

# Solo lanzar interfaz gráfica
python run_system.py --gui
```

## 📊 Componentes del Sistema

### 1. Indicadores Predictivos (`Features/predictive_indicators.py`)

**Función**: Genera métricas detalladas de éxito por género y plataforma.

**Indicadores Generados**:
- Probabilidad de éxito promedio por género
- Top plataformas por probabilidad de éxito
- Combinaciones exitosas género-plataforma
- Recomendaciones de inversión
- Análisis de saturación del mercado

**Archivos de Salida**:
- `Data/Processed/reporte_indicadores_predictivos.md`
- `Data/Processed/indicadores_predictivos.png`
- `Data/Processed/genre_indicators.csv`
- `Data/Processed/platform_indicators.csv`

**Ejemplo de Uso**:
```python
from Features.predictive_indicators import PredictiveIndicators

analyzer = PredictiveIndicators()
genre_indicators, platform_indicators, insights, results_df = analyzer.run_complete_analysis()
```

### 2. Interfaz Gráfica (`GUI/game_predictor_gui.py`)

**Función**: Interfaz intuitiva para realizar predicciones y visualizar análisis.

**Características**:
- **Pestaña de Predicción Individual**: Ingreso de características de juego para predicción inmediata
- **Pestaña de Análisis General**: Visualización de indicadores y métricas del sistema
- **Pestaña de Indicadores Visuales**: Gráficos interactivos de rendimiento por género/plataforma

**Funcionalidades Principales**:
- **🔍 Búsqueda por Nombre**: Ingresa el nombre de un juego real y el sistema autocompleta todos los campos
- **📊 Análisis Automático**: Clasificación inmediata con explicación detallada de factores
- **⚙️ Entrada Manual**: Opción de ingresar características personalizadas

**Campos de Entrada**:
- **Búsqueda de Juego**: Nombre del juego para búsqueda automática en la base de datos
- Género del juego (Action, Adventure, Sports, etc.)
- Plataforma de lanzamiento (PS4, Xbox, PC, etc.)
- Puntuación de crítica (0-100)
- Puntuación de usuario (0-10)
- Año de lanzamiento

**Ejemplo de Uso**:
```bash
python run_system.py --gui
```

**🔍 Funcionalidad de Búsqueda de Juegos**:

La interfaz permite buscar juegos reales de la base de datos:

1. **Búsqueda Automática**: Escriba parte del nombre del juego (ej: "mario", "fifa", "cod")
2. **Selección Múltiple**: Si hay varios resultados, aparece una ventana de selección
3. **Autocompletado**: El sistema llena automáticamente todos los campos con datos reales
4. **Análisis Inmediato**: Genera predicción y explicación detallada automáticamente

**Ejemplos de Búsquedas Exitosas**:
- `"mario"` → Super Mario Bros, Super Mario World, etc.
- `"call of duty"` → Call of Duty series
- `"fifa"` → FIFA series
- `"pokemon"` → Pokemon series
- `"zelda"` → The Legend of Zelda series

### 3. Sistema de Validación (`Tests/test_system_validation.py`)

**Función**: Suite completa de pruebas para validar el funcionamiento del sistema.

**Pruebas Implementadas**:

#### Prueba 1: Integridad de Datos
- Verificación de dimensiones features vs labels
- Detección de valores faltantes
- Análisis de distribución de clases
- Detección de valores infinitos

#### Prueba 2: Rendimiento de Modelos
- Cálculo de métricas: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Comparación MLP vs Baseline
- Evaluación de umbrales de rendimiento

#### Prueba 3: Consistencia de Predicciones
- Pruebas de reproducibilidad (múltiples ejecuciones)
- Pruebas de robustez ante ruido
- Análisis de estabilidad del modelo

#### Prueba 4: Rendimiento por Género
- Métricas específicas por cada género
- Identificación de géneros problemáticos
- Análisis de variabilidad entre géneros

#### Prueba 5: Casos Extremos
- Entrada con valores cero
- Entrada con valores muy grandes
- Entrada con valores negativos
- Diferentes tamaños de batch

**Archivos de Salida**:
- `Tests/validation_report.html` - Reporte HTML detallado
- `Tests/validation_plots.png` - Visualizaciones de rendimiento

**Ejemplo de Uso**:
```python
from Tests.test_system_validation import SystemValidationTests

validator = SystemValidationTests()
success = validator.run_all_tests()
```

## 📈 Interpretación de Resultados

### Indicadores de Éxito por Género

El sistema categoriza géneros en tres niveles:

- **🟢 Alto Potencial** (≥70% probabilidad): Géneros con excelentes perspectivas
- **🟡 Potencial Medio** (40-70% probabilidad): Géneros con buenas posibilidades
- **🔴 Bajo Potencial** (<40% probabilidad): Géneros con riesgo elevado

### Métricas de Validación

- **Accuracy ≥ 0.8**: Excelente rendimiento
- **Accuracy 0.6-0.8**: Buen rendimiento
- **Accuracy < 0.6**: Rendimiento requiere mejora

### Interpretación de Predicciones Individuales

La GUI proporciona análisis detallados con múltiples factores:

#### 🔮 Clasificaciones del Sistema:
- **🟢 TOP-SELLER POTENCIAL** (≥70%): Excelentes perspectivas, invertir con confianza
- **🟡 VENTAS MODERADAS** (50-70%): Buenas posibilidades con optimizaciones
- **🔴 RIESGO ELEVADO** (<50%): Precaución, revisar estrategia fundamental

#### 💡 Análisis de Factores Incluidos:
1. **Factor Base del Género**: Probabilidad histórica promedio del género
2. **Factor de Puntuaciones**: Impacto de críticas y usuarios
3. **Factor de Plataforma**: Rendimiento histórico de la plataforma
4. **Factor Temporal**: Ventaja/desventaja por año de lanzamiento
5. **Nivel de Competencia**: Saturación del mercado en el género

#### 📈 Recomendaciones Automáticas:
- **Acciones Sugeridas**: Pasos específicos según la clasificación
- **Comparación con Mercado**: Posición vs promedio del género
- **Alertas Estratégicas**: Advertencias sobre géneros saturados o de bajo rendimiento

## 🔧 Personalización y Extensión

### Agregar Nuevos Géneros
1. Actualizar los datos en `Data/Raw/`
2. Ejecutar `python run_system.py --prepare --train`
3. Regenerar indicadores con `python run_system.py --indicators`

### Modificar Arquitectura del Modelo
Editar `Models/train_pytorch.py`:
```python
# Cambiar arquitectura MLP
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, hidden3=32):
        # Agregar capa adicional
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden3, 1)
        )
```

### Agregar Nuevas Características
Modificar `Features/build_features.py`:
```python
# Ejemplo: agregar interacción año-género
df["year_genre_interaction"] = df["year_clean"].astype(str) + "_" + df["Genre"]
```

## 📊 Análisis de Funcionamiento

### Metodología de Validación

1. **Validación Cruzada**: División estratificada de datos (80% entrenamiento, 20% validación)
2. **Early Stopping**: Detención automática para evitar sobreajuste
3. **Métricas Múltiples**: No solo accuracy, sino precision, recall, F1-score y ROC-AUC
4. **Análisis por Segmentos**: Evaluación específica por género para detectar sesgos
5. **Pruebas de Robustez**: Verificación de estabilidad ante perturbaciones

### Criterios de Éxito

El sistema se considera **exitoso** si cumple:

- ✅ **Accuracy > 75%** en el conjunto de validación
- ✅ **MLP supera al baseline** por al menos 5%
- ✅ **Todas las pruebas de integridad** pasan
- ✅ **Predicciones consistentes** (std < 0.01 en múltiples ejecuciones)
- ✅ **Sin errores en casos extremos**

## 📚 Referencias y Metodología

### Datasets Utilizados
- Video Game Sales with Ratings (Kaggle)
- Popular Video Games 1980-2023 (Kaggle)
- Video Games Reviews (Kaggle)

### Criterio de Éxito
- **Top-Seller**: Juegos en el percentil 75 de ventas dentro de su género
- **Enfoque Relativo**: Comparación dentro del mismo género, no absoluta
- **Validación Cruzada**: Estratificada para mantener distribución de clases

