# Features/predictive_indicators.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import torch
import torch.nn as nn
from Models.train_pytorch import MLPClassifier, PerceptronBaseline
import warnings
warnings.filterwarnings('ignore')

class PredictiveIndicators:
    # clase indicadores predictivos - exito probable de videojuegos x genero

    def __init__(self, model_path="Models/mlp_classifier.pth"):
        self.model_path = model_path
        self.model = None
        self.features_df = None
        self.labels_df = None
        self.original_data = None
        self.predictions = None
        self.probabilities = None
        
    def load_data(self):
        #LE PUSE EMOJIS AAAAAAAAAAA
        print("📊 Cargando datos procesados...")
        
        # cargar datos de entrenamiento
        self.features_df = pd.read_csv("Data/Processed/features_matrix.csv")
        self.labels_df = pd.read_csv("Data/Processed/labels.csv")
        self.original_data = pd.read_csv("Data/Processed/vgsales_integrated_refined.csv")
        
        # cargar modelo entrenado
        input_dim = self.features_df.shape[1]
        self.model = MLPClassifier(input_dim)
        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model.eval()
        
        print(f"✅ Datos cargados: {len(self.features_df)} juegos")
        print(f"✅ Modelo cargado desde: {self.model_path}")
        
    def generate_predictions(self):
        print("📊 Generando predicciones...")
        
        X = self.features_df.values.astype(float)
        
        # tratamiento de valores faltantes
        if np.isnan(X).any():
            col_medias = np.nanmedian(X, axis=0)
            inds_nan = np.where(np.isnan(X))
            X[inds_nan] = np.take(col_medias, inds_nan[1])
            
        if np.isinf(X).any():
            X[np.isinf(X)] = 0.0
        
        X_tensor = torch.from_numpy(X).float()
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            self.probabilities = torch.sigmoid(logits).numpy().flatten()
            self.predictions = (self.probabilities >= 0.5).astype(int)
            
        print(f"✅ Predicciones generadas para {len(self.predictions)} juegos")
    
    def calculate_success_indicators_by_genre(self):
        print("📊 Calculando indicadores de éxito por género...")
        
        # crear dataframe combinado
        results_df = pd.DataFrame({
            'Genre': self.original_data['Genre'],
            'Game_Name': self.original_data['name_clean'],
            'Platform': self.original_data['platform_clean'],
            'Year': self.original_data['year_clean'],
            'Global_Sales': self.original_data['Global_Sales'],
            'Actual_TopSeller': self.labels_df.values.flatten(),
            'Predicted_TopSeller': self.predictions,
            'Success_Probability': self.probabilities
        })
        
        # indicadores por genero
        genre_indicators = results_df.groupby('Genre').agg({
            'Success_Probability': ['mean', 'std', 'min', 'max'],
            'Actual_TopSeller': 'sum',
            'Predicted_TopSeller': 'sum',
            'Global_Sales': ['mean', 'std'],
            'Game_Name': 'count'
        }).round(4)
        
        genre_indicators.columns = [
            'Prob_Promedio', 'Prob_Desv_Est', 'Prob_Minima', 'Prob_Maxima',
            'TopSellers_Reales', 'TopSellers_Predichos', 'Ventas_Promedio', 
            'Ventas_Desv_Est', 'Total_Juegos'
        ]
        
        # calcular metricas adicionales
        genre_indicators['Tasa_Exito_Real'] = (
            genre_indicators['TopSellers_Reales'] / genre_indicators['Total_Juegos']
        ).round(4)
        
        genre_indicators['Tasa_Exito_Predicha'] = (
            genre_indicators['TopSellers_Predichos'] / genre_indicators['Total_Juegos']
        ).round(4)
        
        # categorizar generos por potencial de exito
        genre_indicators['Categoria_Riesgo'] = genre_indicators['Prob_Promedio'].apply(
            lambda x: 'Alto Potencial' if x >= 0.7 
                     else 'Potencial Medio' if x >= 0.4 
                     else 'Bajo Potencial'
        )
        
        return genre_indicators, results_df
    
    def calculate_platform_indicators(self, results_df):
        print("📊 Calculando indicadores por plataforma...")
        
        platform_indicators = results_df.groupby('Platform').agg({
            'Success_Probability': ['mean', 'std'],
            'Actual_TopSeller': 'sum',
            'Global_Sales': 'mean',
            'Game_Name': 'count'
        }).round(4)
        
        platform_indicators.columns = [
            'Prob_Promedio', 'Prob_Desv_Est', 'TopSellers_Reales', 
            'Ventas_Promedio', 'Total_Juegos'
        ]
        
        # solo plataformas con al menos 10 juegos
        platform_indicators = platform_indicators[platform_indicators['Total_Juegos'] >= 10]
        platform_indicators = platform_indicators.sort_values('Prob_Promedio', ascending=False)
        
        return platform_indicators
    
    def generate_market_insights(self, genre_indicators, results_df):
        print("📊 Generando insights de mercado...")
        
        insights = {
            'generos_alto_potencial': genre_indicators[
                genre_indicators['Categoria_Riesgo'] == 'Alto Potencial'
            ].index.tolist(),
            
            'generos_saturados': genre_indicators[
                (genre_indicators['Total_Juegos'] > genre_indicators['Total_Juegos'].quantile(0.75)) &
                (genre_indicators['Prob_Promedio'] < 0.5)
            ].index.tolist(),
            
            'combinaciones_exitosas': self._find_successful_combinations(results_df),
            
            'tendencias_temporales': self._analyze_temporal_trends(results_df),
            
            'recomendaciones_inversion': self._generate_investment_recommendations(genre_indicators)
        }
        
        return insights
    
    def _find_successful_combinations(self, results_df):
        combo_analysis = results_df.groupby(['Genre', 'Platform']).agg({
            'Success_Probability': 'mean',
            'Game_Name': 'count'
        }).reset_index()
        
        # solo combinaciones con al menos 5 juegos
        combo_analysis = combo_analysis[combo_analysis['Game_Name'] >= 5]
        combo_analysis = combo_analysis.sort_values('Success_Probability', ascending=False)
        
        return combo_analysis.head(10).to_dict('records')
    
    def _analyze_temporal_trends(self, results_df):
        temporal_trends = results_df.groupby('Year').agg({
            'Success_Probability': 'mean',
            'Global_Sales': 'mean',
            'Game_Name': 'count'
        }).reset_index()

        temporal_trends = temporal_trends[temporal_trends['Game_Name'] >= 10]
        
        return temporal_trends.to_dict('records')
    
    def _generate_investment_recommendations(self, genre_indicators):
        recommendations = []
        
        # ordenar por potencial vs saturacion
        genre_indicators['investment_score'] = (
            genre_indicators['Prob_Promedio'] * 0.6 + 
            (1 / (genre_indicators['Total_Juegos'] / 100)) * 0.4
        )
        
        top_genres = genre_indicators.sort_values('investment_score', ascending=False).head(5)
        
        for genre in top_genres.index:
            row = genre_indicators.loc[genre]
            recommendations.append({
                'genero': genre,
                'probabilidad_exito': row['Prob_Promedio'],
                'competencia': 'Baja' if row['Total_Juegos'] < 50 else 'Media' if row['Total_Juegos'] < 200 else 'Alta',
                'ventas_promedio': row['Ventas_Promedio'],
                'recomendacion': f"Género con {row['Prob_Promedio']:.1%} de probabilidad de éxito"
            })
        
        return recommendations
    
    def create_visualizations(self, genre_indicators, platform_indicators, results_df):
        print("📊 Creando visualizaciones...")
        
        # configurar estilo
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Indicadores Predictivos de Éxito en Videojuegos', fontsize=16, fontweight='bold')
        
        # 1. probabilidad de exito por genero
        ax1 = axes[0, 0]
        genre_prob = genre_indicators['Prob_Promedio'].sort_values(ascending=True)
        colors = ['#ff6b6b' if x < 0.4 else '#ffd93d' if x < 0.7 else '#6bcf7f' for x in genre_prob.values]
        genre_prob.plot(kind='barh', ax=ax1, color=colors)
        ax1.set_title('Probabilidad de Éxito por Género', fontweight='bold')
        ax1.set_xlabel('Probabilidad Promedio')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. distribucion de probabilidades
        ax2 = axes[0, 1]
        ax2.hist(results_df['Success_Probability'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(results_df['Success_Probability'].mean(), color='red', linestyle='--', 
                   label=f'Media: {results_df["Success_Probability"].mean():.3f}')
        ax2.set_title('Distribución de Probabilidades de Éxito', fontweight='bold')
        ax2.set_xlabel('Probabilidad de Éxito')
        ax2.set_ylabel('Frecuencia')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. top plataformas por probabilidad
        ax3 = axes[1, 0]
        top_platforms = platform_indicators.head(10)['Prob_Promedio']
        top_platforms.plot(kind='bar', ax=ax3, color='lightcoral')
        ax3.set_title('Top 10 Plataformas por Probabilidad de Éxito', fontweight='bold')
        ax3.set_ylabel('Probabilidad Promedio')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. relacion entre ventas reales y probabilidad predicha
        ax4 = axes[1, 1]
        scatter_data = results_df.sample(min(1000, len(results_df)))  # Muestra para legibilidad
        colors = ['red' if x == 1 else 'blue' for x in scatter_data['Actual_TopSeller']]
        ax4.scatter(scatter_data['Global_Sales'], scatter_data['Success_Probability'], 
                   c=colors, alpha=0.6, s=20)
        ax4.set_title('Ventas Reales vs Probabilidad Predicha', fontweight='bold')
        ax4.set_xlabel('Ventas Globales (Millones)')
        ax4.set_ylabel('Probabilidad de Éxito')
        ax4.grid(alpha=0.3)
        
        # leyenda 
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                markersize=8, label='Top-Seller Real'),
                         Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                markersize=8, label='No Top-Seller')]
        ax4.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('Data/Processed/indicadores_predictivos.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizaciones guardadas en: Data/Processed/indicadores_predictivos.png")
    
    def generate_detailed_report(self, genre_indicators, platform_indicators, insights):
        print("📊 Generando reporte detallado...")
        
        report = f"""
// REPORTE DE INDICADORES PREDICTIVOS DE ÉXITO EN VIDEOJUEGOS

**Métricas Generales:**
- Total de juegos analizados: {len(self.original_data):,}
- Géneros evaluados: {len(genre_indicators)}
- Plataformas evaluadas: {len(platform_indicators)}
- Precisión promedio del modelo: {np.mean(self.predictions == self.labels_df.values.flatten()):.1%}

// INDICADORES POR GÉNERO

/// Géneros de Alto Potencial (Prob. Éxito > 70%)
"""
        
        high_potential = genre_indicators[genre_indicators['Categoria_Riesgo'] == 'Alto Potencial']
        for genre in high_potential.index:
            row = high_potential.loc[genre]
            report += f"""
**{genre}:**
- Probabilidad de éxito: {row['Prob_Promedio']:.1%}
- Juegos analizados: {row['Total_Juegos']}
- Ventas promedio: ${row['Ventas_Promedio']:.2f}M
- Tasa de éxito real: {row['Tasa_Exito_Real']:.1%}
"""

        report += f"""
/// Géneros de Potencial Medio (40% - 70%)
"""
        medium_potential = genre_indicators[genre_indicators['Categoria_Riesgo'] == 'Potencial Medio']
        for genre in medium_potential.index[:5]:  # Top 5
            row = medium_potential.loc[genre]
            report += f"- **{genre}**: {row['Prob_Promedio']:.1%} prob. éxito, {row['Total_Juegos']} juegos\n"

        report += f"""
// INDICADORES POR PLATAFORMA

/// Top 5 Plataformas por Probabilidad de Éxito:
"""
        for i, (platform, row) in enumerate(platform_indicators.head(5).iterrows(), 1):
            report += f"{i}. **{platform}**: {row['Prob_Promedio']:.1%} ({row['Total_Juegos']} juegos)\n"

        report += f"""
// MERCADO

/// Combinaciones Género-Plataforma Más Exitosas:
"""
        for combo in insights['combinaciones_exitosas'][:5]:
            report += f"- {combo['Genre']} en {combo['Platform']}: {combo['Success_Probability']:.1%}\n"

        report += f"""
/// Recomendaciones de Inversión:
"""
        for rec in insights['recomendaciones_inversion']:
            report += f"""
**{rec['genero']}:**
- Probabilidad de éxito: {rec['probabilidad_exito']:.1%}
- Nivel de competencia: {rec['competencia']}
- Ventas promedio esperadas: ${rec['ventas_promedio']:.2f}M
- {rec['recomendacion']}
"""

        report += f"""
## FACTORES DE EXITO

### Características que más impactan el éxito:
1. **Género del juego**: Los géneros de acción y aventura muestran mayor probabilidad de éxito
2. **Plataforma de lanzamiento**: Las plataformas mainstream tienen mejor performance
3. **Timing de lanzamiento**: Años con menos saturación del mercado
4. **Puntuaciones de crítica**: Correlación positiva con éxito comercial

## LIMITACIONES / CONSIDERACIONES DEL MODELO

- Los indicadores se basan en datos históricos y pueden no reflejar tendencias futuras
- Se considera el percentil 75 de ventas por género como posible éxito
- Factores externos como marketing y eventos del mercado no están incluidos

"""
        
        # Guardar reporte
        with open('Data/Processed/reporte_indicadores_predictivos.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("✅ Reporte guardado en: Data/Processed/reporte_indicadores_predictivos.md")
        return report
    
    def run_complete_analysis(self):
        print("🚀 Iniciando análisis completo de indicadores predictivos...\n")
        
        # cargar datos y generar predicciones
        self.load_data()
        self.generate_predictions()
        
        # calcular indicadores
        genre_indicators, results_df = self.calculate_success_indicators_by_genre()
        platform_indicators = self.calculate_platform_indicators(results_df)
        insights = self.generate_market_insights(genre_indicators, results_df)
        
        # mostrar resumen en consola
        print("\n" + "="*60)
        print("RESUMEN DE INDICADORES PREDICTIVOS")
        print("="*60)
        
        print(f"\n📊 GÉNEROS DE ALTO POTENCIAL:")
        high_potential = genre_indicators[genre_indicators['Categoria_Riesgo'] == 'Alto Potencial']
        for genre in high_potential.index:
            prob = high_potential.loc[genre, 'Prob_Promedio']
            print(f"   • {genre}: {prob:.1%} probabilidad de éxito")
        
        print(f"\n🎮 TOP 3 PLATAFORMAS:")
        for i, (platform, row) in enumerate(platform_indicators.head(3).iterrows(), 1):
            print(f"   {i}. {platform}: {row['Prob_Promedio']:.1%}")
        
        print(f"\n📊 INSIGHTS CLAVE:")
        print(f"   • Géneros saturados: {', '.join(insights['generos_saturados'][:3])}")
        print(f"   • Mejor combinación: {insights['combinaciones_exitosas'][0]['Genre']} en {insights['combinaciones_exitosas'][0]['Platform']}")
        
        # crear visualizaciones y reporte
        self.create_visualizations(genre_indicators, platform_indicators, results_df)
        report = self.generate_detailed_report(genre_indicators, platform_indicators, insights)
        
        # guardar datos para GUI
        genre_indicators.to_csv('Data/Processed/genre_indicators.csv')
        platform_indicators.to_csv('Data/Processed/platform_indicators.csv')
        results_df.to_csv('Data/Processed/prediction_results.csv', index=False)
        
        print("\n✅ Análisis completo finalizado!")
        print("Archivos generados:")
        print("   • Data/Processed/indicadores_predictivos.png")
        print("   • Data/Processed/reporte_indicadores_predictivos.md")
        print("   • Data/Processed/genre_indicators.csv")
        print("   • Data/Processed/platform_indicators.csv")
        print("   • Data/Processed/prediction_results.csv")
        
        return genre_indicators, platform_indicators, insights, results_df

if __name__ == "__main__":
    analyzer = PredictiveIndicators()
    analyzer.run_complete_analysis()