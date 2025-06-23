# Tests/test_system_validation.py

import unittest
import pandas as pd
import numpy as np
import torch
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.train_pytorch import MLPClassifier, PerceptronBaseline
from Features.predictive_indicators import PredictiveIndicators
from Features.build_features import build_and_save_features

class SystemValidationTests:   
    def __init__(self):
        self.test_results = {}
        self.models = {}
        self.data = {}
        self.report_path = "Tests/validation_report.html"
        
    def setup_test_environment(self):
        print("🔧 Configurando entorno de pruebas...")
        
        try:
            self.data['features'] = pd.read_csv("Data/Processed/features_matrix.csv")
            self.data['labels'] = pd.read_csv("Data/Processed/labels.csv")
            self.data['original'] = pd.read_csv("Data/Processed/vgsales_integrated_refined.csv")

            input_dim = self.data['features'].shape[1]
            
            self.models['mlp'] = MLPClassifier(input_dim)
            if os.path.exists("Models/mlp_classifier.pth"):
                self.models['mlp'].load_state_dict(torch.load("Models/mlp_classifier.pth", map_location='cpu'))
                self.models['mlp'].eval()
            
            self.models['baseline'] = PerceptronBaseline(input_dim)
            if os.path.exists("Models/baseline_perceptron.pth"):
                self.models['baseline'].load_state_dict(torch.load("Models/baseline_perceptron.pth", map_location='cpu'))
                self.models['baseline'].eval()
            
            print("✅ Entorno configurado correctamente")
            return True
            
        except Exception as e:
            print(f"❌ Error configurando entorno: {str(e)}")
            return False
    
    def test_data_integrity(self):
        print("\n📊 Ejecutando Prueba 1: Integridad de Datos")
        
        test_name = "data_integrity"
        results = {
            'name': 'Integridad de Datos',
            'status': 'PASSED',
            'details': [],
            'warnings': [],
            'errors': []
        }
        
        features_shape = self.data['features'].shape
        labels_shape = self.data['labels'].shape
        
        if features_shape[0] != labels_shape[0]:
            results['errors'].append(f"Mismatch en número de filas: Features {features_shape[0]} vs Labels {labels_shape[0]}")
            results['status'] = 'FAILED'
        else:
            results['details'].append(f"✅ Dimensiones correctas: {features_shape[0]} muestras, {features_shape[1]} características")
        
        features_nan = self.data['features'].isnull().sum().sum()
        labels_nan = self.data['labels'].isnull().sum().sum()
        
        if features_nan > 0:
            results['warnings'].append(f"Valores faltantes en features: {features_nan}")
        if labels_nan > 0:
            results['errors'].append(f"Valores faltantes en labels: {labels_nan}")
            results['status'] = 'FAILED'
        else:
            results['details'].append("✅ No hay valores faltantes en labels")
        
        class_distribution = self.data['labels'].value_counts()
        minority_class_ratio = min(class_distribution) / len(self.data['labels'])
        
        if minority_class_ratio < 0.1:
            results['warnings'].append(f"Dataset desbalanceado: {minority_class_ratio:.1%} clase minoritaria")
        else:
            results['details'].append(f"✅ Distribución balanceada: {minority_class_ratio:.1%} clase minoritaria")
        
        feature_stats = self.data['features'].describe()
        infinite_values = np.isinf(self.data['features'].values).sum()
        
        if infinite_values > 0:
            results['errors'].append(f"Valores infinitos encontrados: {infinite_values}")
            results['status'] = 'FAILED'
        else:
            results['details'].append("✅ No hay valores infinitos en features")
        
        self.test_results[test_name] = results
        print(f"   Status: {results['status']}")
        return results['status'] == 'PASSED'
    
    def test_model_performance(self):
        print("\n📊 Ejecutando Prueba 2: Rendimiento de Modelos")
        
        test_name = "model_performance"
        results = {
            'name': 'Rendimiento de Modelos',
            'status': 'PASSED',
            'details': [],
            'warnings': [],
            'errors': [],
            'metrics': {}
        }
        
        try:
            X = self.data['features'].values.astype(float)
            y = self.data['labels'].values.flatten()
            
            if np.isnan(X).any():
                col_medias = np.nanmedian(X, axis=0)
                inds_nan = np.where(np.isnan(X))
                X[inds_nan] = np.take(col_medias, inds_nan[1])
            
            X_tensor = torch.from_numpy(X).float()
            
            for model_name, model in self.models.items():
                if model is None:
                    results['warnings'].append(f"Modelo {model_name} no cargado")
                    continue
                
                with torch.no_grad():
                    logits = model(X_tensor)
                    probabilities = torch.sigmoid(logits).numpy().flatten()
                    predictions = (probabilities >= 0.5).astype(int)
                
                accuracy = accuracy_score(y, predictions)
                precision = precision_score(y, predictions, average='weighted', zero_division=0)
                recall = recall_score(y, predictions, average='weighted', zero_division=0)
                f1 = f1_score(y, predictions, average='weighted', zero_division=0)
                
                try:
                    if len(np.unique(y)) == 2:
                        roc_auc = roc_auc_score(y, probabilities)
                    else:
                        roc_auc = None
                except:
                    roc_auc = None
                
                results['metrics'][model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc
                }
                
                if accuracy < 0.6:
                    results['warnings'].append(f"{model_name}: Precisión baja ({accuracy:.3f})")
                elif accuracy >= 0.8:
                    results['details'].append(f"✅ {model_name}: Excelente precisión ({accuracy:.3f})")
                else:
                    results['details'].append(f"✅ {model_name}: Buena precisión ({accuracy:.3f})")
                
                if f1 < 0.6:
                    results['warnings'].append(f"{model_name}: F1-Score bajo ({f1:.3f})")
            
            if 'mlp' in results['metrics'] and 'baseline' in results['metrics']:
                mlp_acc = results['metrics']['mlp']['accuracy']
                baseline_acc = results['metrics']['baseline']['accuracy']
                
                if mlp_acc > baseline_acc + 0.05:
                    results['details'].append(f"✅ MLP supera significativamente al baseline (+{mlp_acc-baseline_acc:.3f})")
                elif mlp_acc > baseline_acc:
                    results['details'].append(f"✅ MLP supera al baseline (+{mlp_acc-baseline_acc:.3f})")
                else:
                    results['warnings'].append(f"MLP no supera al baseline ({mlp_acc:.3f} vs {baseline_acc:.3f})")
            
        except Exception as e:
            results['errors'].append(f"Error evaluando modelos: {str(e)}")
            results['status'] = 'FAILED'
        
        self.test_results[test_name] = results
        print(f"   Status: {results['status']}")
        return results['status'] == 'PASSED'
    
    def test_prediction_consistency(self):
        print("\n🔄 Ejecutando Prueba 3: Consistencia de Predicciones")
        
        test_name = "prediction_consistency"
        results = {
            'name': 'Consistencia de Predicciones',
            'status': 'PASSED',
            'details': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            sample_size = min(100, len(self.data['features']))
            sample_indices = np.random.choice(len(self.data['features']), sample_size, replace=False)
            
            X_sample = self.data['features'].iloc[sample_indices].values.astype(float)
            
            if np.isnan(X_sample).any():
                col_medias = np.nanmedian(X_sample, axis=0)
                inds_nan = np.where(np.isnan(X_sample))
                X_sample[inds_nan] = np.take(col_medias, inds_nan[1])
            
            X_tensor = torch.from_numpy(X_sample).float()
            
            for model_name, model in self.models.items():
                if model is None:
                    continue
                
                predictions_list = []
                
                for _ in range(5):
                    with torch.no_grad():
                        logits = model(X_tensor)
                        probabilities = torch.sigmoid(logits).numpy().flatten()
                        predictions_list.append(probabilities)
                
                predictions_array = np.array(predictions_list)
                std_predictions = np.std(predictions_array, axis=0)
                max_std = np.max(std_predictions)
                
                if max_std > 0.01:
                    results['warnings'].append(f"{model_name}: Predicciones inconsistentes (std máx: {max_std:.4f})")
                else:
                    results['details'].append(f"✅ {model_name}: Predicciones consistentes (std máx: {max_std:.4f})")
                
            if 'mlp' in self.models and self.models['mlp'] is not None:
                noise_level = 0.01
                X_noisy = X_sample + np.random.normal(0, noise_level, X_sample.shape)
                X_noisy_tensor = torch.from_numpy(X_noisy).float()
                
                with torch.no_grad():
                    original_probs = torch.sigmoid(self.models['mlp'](X_tensor)).numpy().flatten()
                    noisy_probs = torch.sigmoid(self.models['mlp'](X_noisy_tensor)).numpy().flatten()
                
                prob_diff = np.abs(original_probs - noisy_probs)
                max_diff = np.max(prob_diff)
                mean_diff = np.mean(prob_diff)
                
                if max_diff > 0.1:
                    results['warnings'].append(f"MLP sensible al ruido (diff máx: {max_diff:.4f})")
                else:
                    results['details'].append(f"✅ MLP robusto al ruido (diff máx: {max_diff:.4f})")
                
        except Exception as e:
            results['errors'].append(f"Error en prueba de consistencia: {str(e)}")
            results['status'] = 'FAILED'
        
        self.test_results[test_name] = results
        print(f"   Status: {results['status']}")
        return results['status'] == 'PASSED'
    
    def test_genre_specific_performance(self):
        print("\n🎮 Ejecutando Prueba 4: Rendimiento por Género")
        
        test_name = "genre_performance"
        results = {
            'name': 'Rendimiento por Género',
            'status': 'PASSED',
            'details': [],
            'warnings': [],
            'errors': [],
            'genre_metrics': {}
        }
        
        try:
            if 'mlp' not in self.models or self.models['mlp'] is None:
                results['errors'].append("Modelo MLP no disponible")
                results['status'] = 'FAILED'
                self.test_results[test_name] = results
                return False
            
            X = self.data['features'].values.astype(float)
            y = self.data['labels'].values.flatten()
            
            if np.isnan(X).any():
                col_medias = np.nanmedian(X, axis=0)
                inds_nan = np.where(np.isnan(X))
                X[inds_nan] = np.take(col_medias, inds_nan[1])
            
            X_tensor = torch.from_numpy(X).float()
            
            with torch.no_grad():
                logits = self.models['mlp'](X_tensor)
                probabilities = torch.sigmoid(logits).numpy().flatten()
                predictions = (probabilities >= 0.5).astype(int)
            
            results_df = pd.DataFrame({
                'Genre': self.data['original']['Genre'],
                'Actual': y,
                'Predicted': predictions,
                'Probability': probabilities
            })
            
            genre_stats = []
            for genre in results_df['Genre'].unique():
                genre_data = results_df[results_df['Genre'] == genre]
                
                if len(genre_data) < 5:
                    continue
                
                accuracy = accuracy_score(genre_data['Actual'], genre_data['Predicted'])
                precision = precision_score(genre_data['Actual'], genre_data['Predicted'], 
                                          average='weighted', zero_division=0)
                recall = recall_score(genre_data['Actual'], genre_data['Predicted'], 
                                    average='weighted', zero_division=0)
                f1 = f1_score(genre_data['Actual'], genre_data['Predicted'], 
                             average='weighted', zero_division=0)
                
                genre_stats.append({
                    'genre': genre,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'samples': len(genre_data)
                })
                
                results['genre_metrics'][genre] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'samples': len(genre_data)
                }
                
                if accuracy < 0.5:
                    results['warnings'].append(f"{genre}: Rendimiento pobre (acc: {accuracy:.3f})")
                elif accuracy >= 0.8:
                    results['details'].append(f"✅ {genre}: Excelente rendimiento (acc: {accuracy:.3f})")
            
            genre_df = pd.DataFrame(genre_stats)
            if not genre_df.empty:
                avg_accuracy = genre_df['accuracy'].mean()
                std_accuracy = genre_df['accuracy'].std()
                min_accuracy = genre_df['accuracy'].min()
                max_accuracy = genre_df['accuracy'].max()
                
                results['details'].append(f"✅ Precisión promedio por género: {avg_accuracy:.3f} (±{std_accuracy:.3f})")
                results['details'].append(f"✅ Rango de precisión: {min_accuracy:.3f} - {max_accuracy:.3f}")
                
                poor_genres = genre_df[genre_df['accuracy'] < 0.6]['genre'].tolist()
                if poor_genres:
                    results['warnings'].append(f"Géneros con bajo rendimiento: {', '.join(poor_genres)}")
                
        except Exception as e:
            results['errors'].append(f"Error en análisis por género: {str(e)}")
            results['status'] = 'FAILED'
        
        self.test_results[test_name] = results
        print(f"   Status: {results['status']}")
        return results['status'] == 'PASSED'
    
    def test_edge_cases(self):
        print("\n⚡ Ejecutando Prueba 5: Casos Extremos")
        
        test_name = "edge_cases"
        results = {
            'name': 'Casos Extremos',
            'status': 'PASSED',
            'details': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            if 'mlp' not in self.models or self.models['mlp'] is None:
                results['errors'].append("Modelo MLP no disponible")
                results['status'] = 'FAILED'
                self.test_results[test_name] = results
                return False
            
            zero_input = torch.zeros(1, self.data['features'].shape[1])
            with torch.no_grad():
                zero_pred = torch.sigmoid(self.models['mlp'](zero_input)).item()
            
            if 0.1 <= zero_pred <= 0.9:
                results['details'].append(f"✅ Entrada ceros: Predicción razonable ({zero_pred:.3f})")
            else:
                results['warnings'].append(f"Entrada ceros: Predicción extrema ({zero_pred:.3f})")
            
            large_input = torch.ones(1, self.data['features'].shape[1]) * 100
            with torch.no_grad():
                large_pred = torch.sigmoid(self.models['mlp'](large_input)).item()
            
            if not np.isnan(large_pred) and not np.isinf(large_pred):
                results['details'].append(f"✅ Entrada grande: Sin overflow ({large_pred:.3f})")
            else:
                results['errors'].append(f"Entrada grande: Overflow/NaN ({large_pred})")
                results['status'] = 'FAILED'
            
            negative_input = torch.ones(1, self.data['features'].shape[1]) * -10
            with torch.no_grad():
                negative_pred = torch.sigmoid(self.models['mlp'](negative_input)).item()
            
            if not np.isnan(negative_pred) and not np.isinf(negative_pred):
                results['details'].append(f"✅ Entrada negativa: Predicción estable ({negative_pred:.3f})")
            else:
                results['errors'].append(f"Entrada negativa: Predicción inválida ({negative_pred})")
                results['status'] = 'FAILED'
            
            for batch_size in [1, 10, 100]:
                test_batch = torch.randn(batch_size, self.data['features'].shape[1])
                with torch.no_grad():
                    batch_pred = torch.sigmoid(self.models['mlp'](test_batch))
                
                if batch_pred.shape[0] == batch_size and not torch.isnan(batch_pred).any():
                    results['details'].append(f"✅ Batch size {batch_size}: OK")
                else:
                    results['errors'].append(f"Batch size {batch_size}: Error en dimensiones o NaN")
                    results['status'] = 'FAILED'
            
        except Exception as e:
            results['errors'].append(f"Error en casos extremos: {str(e)}")
            results['status'] = 'FAILED'
        
        self.test_results[test_name] = results
        print(f"   Status: {results['status']}")
        return results['status'] == 'PASSED'
    
    def generate_performance_visualizations(self):
        print("\n📊 Generando visualizaciones de rendimiento...")
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Reporte de Validación del Sistema de Predicción', fontsize=16, fontweight='bold')
            
            if 'model_performance' in self.test_results:
                metrics_data = self.test_results['model_performance']['metrics']
                
                if metrics_data:
                    models = list(metrics_data.keys())
                    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
                    
                    x = np.arange(len(metrics))
                    width = 0.35
                    
                    for i, model in enumerate(models):
                        values = [metrics_data[model][metric] for metric in metrics]
                        axes[0, 0].bar(x + i*width, values, width, label=model)
                    
                    axes[0, 0].set_title('Comparación de Métricas por Modelo')
                    axes[0, 0].set_xlabel('Métricas')
                    axes[0, 0].set_ylabel('Valor')
                    axes[0, 0].set_xticks(x + width/2)
                    axes[0, 0].set_xticklabels(metrics, rotation=45)
                    axes[0, 0].legend()
                    axes[0, 0].grid(axis='y', alpha=0.3)
            
            if 'genre_performance' in self.test_results:
                genre_data = self.test_results['genre_performance']['genre_metrics']
                
                if genre_data:
                    genres = list(genre_data.keys())
                    accuracies = [genre_data[genre]['accuracy'] for genre in genres]
                    
                    colors = ['#2ecc71' if acc >= 0.8 else '#f39c12' if acc >= 0.6 else '#e74c3c' for acc in accuracies]
                    
                    axes[0, 1].bar(range(len(genres)), accuracies, color=colors)
                    axes[0, 1].set_title('Precisión por Género')
                    axes[0, 1].set_xlabel('Géneros')
                    axes[0, 1].set_ylabel('Precisión')
                    axes[0, 1].set_xticks(range(len(genres)))
                    axes[0, 1].set_xticklabels(genres, rotation=45, ha='right')
                    axes[0, 1].grid(axis='y', alpha=0.3)
            
            if self.data['labels'] is not None:
                class_counts = self.data['labels'].value_counts()
                axes[0, 2].pie(class_counts.values, labels=['No Top-Seller', 'Top-Seller'], 
                              autopct='%1.1f%%', startangle=90)
                axes[0, 2].set_title('Distribución de Clases')
            
            if 'model_performance' in self.test_results and self.data['labels'] is not None:
                X = self.data['features'].values.astype(float)
                y = self.data['labels'].values.flatten()
                
                if np.isnan(X).any():
                    col_medias = np.nanmedian(X, axis=0)
                    inds_nan = np.where(np.isnan(X))
                    X[inds_nan] = np.take(col_medias, inds_nan[1])
                
                if 'mlp' in self.models and self.models['mlp'] is not None:
                    X_tensor = torch.from_numpy(X).float()
                    with torch.no_grad():
                        logits = self.models['mlp'](X_tensor)
                        predictions = (torch.sigmoid(logits).numpy().flatten() >= 0.5).astype(int)
                    
                    cm = confusion_matrix(y, predictions)
                    sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
                    axes[1, 0].set_title('Matriz de Confusión - MLP')
                    axes[1, 0].set_xlabel('Predicción')
                    axes[1, 0].set_ylabel('Real')
            
            test_status = []
            test_names = []
            
            for test_name, result in self.test_results.items():
                test_names.append(result['name'])
                test_status.append(1 if result['status'] == 'PASSED' else 0)
            
            colors = ['#2ecc71' if status else '#e74c3c' for status in test_status]
            axes[1, 1].bar(range(len(test_names)), test_status, color=colors)
            axes[1, 1].set_title('Estado de las Pruebas')
            axes[1, 1].set_xlabel('Pruebas')
            axes[1, 1].set_ylabel('Estado (1=Pasada, 0=Fallida)')
            axes[1, 1].set_xticks(range(len(test_names)))
            axes[1, 1].set_xticklabels(test_names, rotation=45, ha='right')
            axes[1, 1].set_ylim(0, 1.2)
            
            axes[1, 2].text(0.5, 0.5, 'Análisis de\nRendimiento\nTemporal\n\n(Disponible tras\nmúltiples ejecuciones)', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Rendimiento Temporal')
            
            plt.tight_layout()
            plt.savefig('Tests/validation_plots.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Visualizaciones guardadas en: Tests/validation_plots.png")
            
        except Exception as e:
            print(f"❌ Error generando visualizaciones: {str(e)}")
    
    def generate_html_report(self):
        print("\n📝 Generando reporte HTML...")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASSED')
        failed_tests = total_tests - passed_tests
        
        html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Validación - Sistema Predictor de Videojuegos</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; }}
        .header {{ text-align: center; color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 20px; }}
        .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; padding: 15px; border-radius: 8px; }}
        .passed {{ background-color: #d5f4e6; color: #27ae60; }}
        .failed {{ background-color: #fadbd8; color: #e74c3c; }}
        .warning {{ background-color: #fdebd0; color: #f39c12; }}
        .test-section {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 8px; }}
        .test-header {{ background-color: #ecf0f1; padding: 15px; font-weight: bold; }}
        .test-content {{ padding: 15px; }}
        .status-passed {{ color: #27ae60; font-weight: bold; }}
        .status-failed {{ color: #e74c3c; font-weight: bold; }}
        .details {{ margin: 10px 0; }}
        .details ul {{ margin: 5px 0; padding-left: 20px; }}
        .timestamp {{ text-align: center; color: #7f8c8d; margin-top: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 REPORTE DE VALIDACIÓN DEL SISTEMA</h1>
            <h2>Predictor de Éxito en Videojuegos</h2>
        </div>
        
        <div class="summary">
            <div class="metric passed">
                <h3>{passed_tests}</h3>
                <p>Pruebas Pasadas</p>
            </div>
            <div class="metric failed">
                <h3>{failed_tests}</h3>
                <p>Pruebas Fallidas</p>
            </div>
            <div class="metric warning">
                <h3>{total_tests}</h3>
                <p>Total de Pruebas</p>
            </div>
        </div>
"""
        
        for test_name, result in self.test_results.items():
            status_class = "status-passed" if result['status'] == 'PASSED' else "status-failed"
            
            html_content += f"""
        <div class="test-section">
            <div class="test-header">
                {result['name']} - <span class="{status_class}">{result['status']}</span>
            </div>
            <div class="test-content">
"""
            
            if result['details']:
                html_content += "<h4>✅ Detalles:</h4><ul>"
                for detail in result['details']:
                    html_content += f"<li>{detail}</li>"
                html_content += "</ul>"
            
            if result['warnings']:
                html_content += "<h4>⚠️ Advertencias:</h4><ul>"
                for warning in result['warnings']:
                    html_content += f"<li>{warning}</li>"
                html_content += "</ul>"
            
            if result['errors']:
                html_content += "<h4>❌ Errores:</h4><ul>"
                for error in result['errors']:
                    html_content += f"<li>{error}</li>"
                html_content += "</ul>"
            
            if 'metrics' in result and result['metrics']:
                html_content += "<h4>📊 Métricas de Rendimiento:</h4>"
                html_content += "<table><tr><th>Modelo</th><th>Precisión</th><th>Recall</th><th>F1-Score</th><th>ROC-AUC</th></tr>"
                
                for model_name, metrics in result['metrics'].items():
                    roc_auc = metrics.get('roc_auc', 'N/A')
                    roc_auc_str = f"{roc_auc:.3f}" if roc_auc is not None else "N/A"
                    
                    html_content += f"""
                    <tr>
                        <td>{model_name}</td>
                        <td>{metrics['accuracy']:.3f}</td>
                        <td>{metrics['recall']:.3f}</td>
                        <td>{metrics['f1_score']:.3f}</td>
                        <td>{roc_auc_str}</td>
                    </tr>
"""
                html_content += "</table>"
            
            if 'genre_metrics' in result and result['genre_metrics']:
                html_content += "<h4>🎮 Rendimiento por Género:</h4>"
                html_content += "<table><tr><th>Género</th><th>Precisión</th><th>F1-Score</th><th>Muestras</th></tr>"
                
                for genre, metrics in result['genre_metrics'].items():
                    html_content += f"""
                    <tr>
                        <td>{genre}</td>
                        <td>{metrics['accuracy']:.3f}</td>
                        <td>{metrics['f1_score']:.3f}</td>
                        <td>{metrics['samples']}</td>
                    </tr>
"""
                html_content += "</table>"
            
            html_content += "</div></div>"
        
        html_content += f"""
        <div class="timestamp">
            <p>Reporte generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Sistema de Validación v1.0 - Proyecto de Predicción de Videojuegos</p>
        </div>
    </div>
</body>
</html>
"""
        
        os.makedirs('Tests', exist_ok=True)
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Reporte HTML guardado en: {self.report_path}")
    
    def run_all_tests(self):
        print("🚀 INICIANDO VALIDACIÓN COMPLETA DEL SISTEMA")
        print("=" * 60)
        
        if not self.setup_test_environment():
            print("❌ Error configurando entorno. Abortando pruebas.")
            return False
        
        test_functions = [
            self.test_data_integrity,
            self.test_model_performance,
            self.test_prediction_consistency,
            self.test_genre_specific_performance,
            self.test_edge_cases
        ]
        
        passed_tests = 0
        total_tests = len(test_functions)
        
        for test_func in test_functions:
            if test_func():
                passed_tests += 1
        
        self.generate_performance_visualizations()
        self.generate_html_report()
        
        print("\n" + "=" * 60)
        print("📋 RESUMEN DE VALIDACIÓN")
        print("=" * 60)
        print(f"Pruebas ejecutadas: {total_tests}")
        print(f"Pruebas pasadas: {passed_tests}")
        print(f"Pruebas fallidas: {total_tests - passed_tests}")
        print(f"Tasa de éxito: {passed_tests/total_tests:.1%}")
        
        if passed_tests == total_tests:
            print("🎉 ¡TODAS LAS PRUEBAS PASARON! El sistema está validado.")
        elif passed_tests >= total_tests * 0.8:
            print("✅ La mayoría de pruebas pasaron. Sistema mayormente validado.")
        else:
            print("⚠️ Varias pruebas fallaron. Revisar errores antes de usar en producción.")
        
        print(f"\n📁 Archivos generados:")
        print(f"   • {self.report_path}")
        print(f"   • Tests/validation_plots.png")
        
        return passed_tests == total_tests

def main():
    validator = SystemValidationTests()
    success = validator.run_all_tests()
    return success

if __name__ == "__main__":
    main()