# GUI/game_predictor_gui.py

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.train_pytorch import MLPClassifier
from Features.predictive_indicators import PredictiveIndicators
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class GamePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎮 Predictor de Éxito en Videojuegos")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # variables
        self.model = None
        self.predictor = PredictiveIndicators()
        self.genre_data = None
        self.platform_data = None
        self.original_data = None
        self.game_index = None
        
        # configurar estilo
        self.setup_style()
        
        # crear interfaz
        self.create_widgets()
        
        # cargar datos iniciales
        self.load_initial_data()
    
    def setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')
    
        style.configure('Title.TLabel', 
                       font=('Helvetica', 16, 'bold'),
                       background='#2c3e50',
                       foreground='#ecf0f1')
        
        style.configure('Header.TLabel',
                       font=('Helvetica', 12, 'bold'),
                       background='#34495e',
                       foreground='#ecf0f1')
        
        style.configure('Custom.TButton',
                       font=('Helvetica', 10, 'bold'),
                       padding=10)
    
    def create_widgets(self):
        # frame principal
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # titulo
        title_label = ttk.Label(main_frame, 
                               text="🎮 SISTEMA PREDICTOR DE ÉXITO EN VIDEOJUEGOS",
                               style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # pestañas
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.create_prediction_tab()
        self.create_analysis_tab()
        self.create_indicators_tab()
        
    def create_prediction_tab(self):
        pred_frame = ttk.Frame(self.notebook)
        self.notebook.add(pred_frame, text="📊 Predicción Individual")
        
        input_frame = tk.Frame(pred_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        ttk.Label(input_frame, text="CARACTERÍSTICAS DEL JUEGO", 
                 style='Header.TLabel').pack(pady=10)
        
        # variables de entrada
        self.game_vars = {}
        
        # campo de busqueda por nombre
        tk.Label(input_frame, text="🔍 Buscar Juego por Nombre:", bg='#34495e', fg='white',
                font=('Helvetica', 10, 'bold')).pack(anchor='w', padx=20, pady=(10,5))
        
        search_frame = tk.Frame(input_frame, bg='#34495e')
        search_frame.pack(fill='x', padx=20, pady=5)
        
        self.game_vars['name_search'] = tk.Entry(search_frame, width=25)
        self.game_vars['name_search'].pack(side='left', fill='x', expand=True)
        
        search_btn = ttk.Button(search_frame, text="🔍 Buscar", 
                               command=self.search_game_by_name)
        search_btn.pack(side='right', padx=(5,0))
        
        # separador
        separator = tk.Frame(input_frame, height=2, bg='#3498db')
        separator.pack(fill='x', padx=20, pady=10)
        
        tk.Label(input_frame, text="⚙️ O ingrese características manualmente:", 
                bg='#34495e', fg='white', font=('Helvetica', 9)).pack(anchor='w', padx=20)
        
        # genero
        tk.Label(input_frame, text="Género:", bg='#34495e', fg='white', 
                font=('Helvetica', 10, 'bold')).pack(anchor='w', padx=20, pady=5)
        self.game_vars['genre'] = ttk.Combobox(input_frame, width=30)
        self.game_vars['genre']['values'] = ['Action', 'Adventure', 'Racing', 'Role-Playing', 
                                           'Simulation', 'Sports', 'Strategy', 'Fighting',
                                           'Platform', 'Puzzle', 'Shooter', 'Misc']
        self.game_vars['genre'].pack(padx=20, pady=5)
        
        # plataforma
        tk.Label(input_frame, text="Plataforma:", bg='#34495e', fg='white',
                font=('Helvetica', 10, 'bold')).pack(anchor='w', padx=20, pady=5)
        self.game_vars['platform'] = ttk.Combobox(input_frame, width=30)
        self.game_vars['platform']['values'] = ['ps4', 'x360', 'ps3', 'wii', 'ds', 
                                               'ps2', 'xb', 'pc', 'gba', 'gc']
        self.game_vars['platform'].pack(padx=20, pady=5)
        
        # puntuacion de critica
        tk.Label(input_frame, text="Puntuación Crítica (0-100):", bg='#34495e', fg='white',
                font=('Helvetica', 10, 'bold')).pack(anchor='w', padx=20, pady=5)
        self.game_vars['critic_score'] = tk.Entry(input_frame, width=32)
        self.game_vars['critic_score'].pack(padx=20, pady=5)
        
        # puntuacion de usuario
        tk.Label(input_frame, text="Puntuación Usuario (0-10):", bg='#34495e', fg='white',
                font=('Helvetica', 10, 'bold')).pack(anchor='w', padx=20, pady=5)
        self.game_vars['user_score'] = tk.Entry(input_frame, width=32)
        self.game_vars['user_score'].pack(padx=20, pady=5)
        
        # anno
        tk.Label(input_frame, text="Año de Lanzamiento:", bg='#34495e', fg='white',
                font=('Helvetica', 10, 'bold')).pack(anchor='w', padx=20, pady=5)
        self.game_vars['year'] = tk.Entry(input_frame, width=32)
        self.game_vars['year'].pack(padx=20, pady=5)
        
        # botones de prediccion
        button_frame = tk.Frame(input_frame, bg='#34495e')
        button_frame.pack(pady=20)
        
        predict_btn = ttk.Button(button_frame, text="🔮 PREDECIR ÉXITO",
                                style='Custom.TButton',
                                command=self.predict_individual_game)
        predict_btn.pack(pady=5)
        
        clear_btn = ttk.Button(button_frame, text="🗑️ LIMPIAR CAMPOS",
                              style='Custom.TButton',
                              command=self.clear_fields)
        clear_btn.pack(pady=5)
        
        # frame derecho para resultados
        result_frame = tk.Frame(pred_frame, bg='#2c3e50', relief=tk.RAISED, bd=2)
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # area de resultados
        ttk.Label(result_frame, text="RESULTADO DE LA PREDICCIÓN", 
                 style='Header.TLabel').pack(pady=10)
        
        self.result_text = tk.Text(result_frame, height=20, width=50, 
                                  bg='#ecf0f1', font=('Courier', 11))
        self.result_text.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
    
    def create_analysis_tab(self):
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="📈 Análisis General")
        
        # botones de control
        control_frame = tk.Frame(analysis_frame, bg='#34495e', height=80)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        control_frame.pack_propagate(False)
        
        ttk.Button(control_frame, text="🔄 Ejecutar Análisis Completo",
                  command=self.run_complete_analysis,
                  style='Custom.TButton').pack(side=tk.LEFT, padx=10, pady=20)
        
        ttk.Button(control_frame, text="📊 Mostrar Indicadores",
                  command=self.show_indicators,
                  style='Custom.TButton').pack(side=tk.LEFT, padx=10, pady=20)
        
        ttk.Button(control_frame, text="💾 Exportar Reporte",
                  command=self.export_report,
                  style='Custom.TButton').pack(side=tk.LEFT, padx=10, pady=20)
        
        # area de visualizacion
        self.analysis_text = tk.Text(analysis_frame, bg='#ecf0f1', 
                                   font=('Courier', 10))
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # scrollbar
        scrollbar = ttk.Scrollbar(self.analysis_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.analysis_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.analysis_text.yview)
    
    def create_indicators_tab(self):
        indicators_frame = ttk.Frame(self.notebook)
        self.notebook.add(indicators_frame, text="📊 Indicadores Visuales")

        self.chart_frame = tk.Frame(indicators_frame, bg='white')
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def load_initial_data(self):
        try:
            if os.path.exists('Data/Processed/genre_indicators.csv'):
                self.genre_data = pd.read_csv('Data/Processed/genre_indicators.csv', index_col=0)
            if os.path.exists('Data/Processed/platform_indicators.csv'):
                self.platform_data = pd.read_csv('Data/Processed/platform_indicators.csv', index_col=0)
            
            if os.path.exists('Data/Processed/vgsales_integrated_refined.csv'):
                self.original_data = pd.read_csv('Data/Processed/vgsales_integrated_refined.csv')
                self.game_index = self.original_data.set_index('name_clean')

            if os.path.exists('Models/mlp_classifier.pth'):
                features_df = pd.read_csv("Data/Processed/features_matrix.csv")
                input_dim = features_df.shape[1]
                self.model = MLPClassifier(input_dim)
                self.model.load_state_dict(torch.load('Models/mlp_classifier.pth', map_location='cpu'))
                self.model.eval()
                
            self.update_status("Sistema cargado correctamente ✅")
        except Exception as e:
            self.update_status(f"Error cargando datos: {str(e)}")
    
    def predict_individual_game(self):
        self.predict_game_with_explanation()
    
    def show_prediction_result(self, genre, platform, critic_score, user_score, year, probability):
        result = f"""
🎮 ANÁLISIS DE PREDICCIÓN DE ÉXITO
{'='*50}

📄 CARACTERÍSTICAS DEL JUEGO:
   • Género: {genre}
   • Plataforma: {platform}
   • Año: {year}
   • Puntuación Crítica: {critic_score}/100
   • Puntuación Usuario: {user_score}/10

📄 RESULTADO DE LA PREDICCIÓN:
   • Probabilidad de Éxito: {probability:.1%}
   • Clasificación: {"TOP-SELLER" if probability >= 0.5 else "VENTAS NORMALES"}

📊 INTERPRETACIÓN:
"""
        
        if probability >= 0.7:
            result += "   🟢 ALTO POTENCIAL - Excelentes perspectivas de éxito"
        elif probability >= 0.5:
            result += "   🟡 POTENCIAL MEDIO - Buenas posibilidades de éxito"
        else:
            result += "   🔴 BAJO POTENCIAL - Riesgo elevado de bajas ventas"
        
        if self.genre_data is not None and genre in self.genre_data.index:
            genre_stats = self.genre_data.loc[genre]
            result += f"""

📊 ESTADÍSTICAS DEL GÉNERO {genre.upper()}:
   • Probabilidad promedio: {genre_stats['Prob_Promedio']:.1%}
   • Juegos analizados: {genre_stats['Total_Juegos']}
   • Ventas promedio: ${genre_stats['Ventas_Promedio']:.2f}M
   • Tasa de éxito real: {genre_stats['Tasa_Exito_Real']:.1%}
"""
        
        result += f"""

📄 RECOMENDACIONES:
   • Evaluar el timing de lanzamiento en el mercado
   • Analizar la competencia en el género seleccionado
   • Considerar estrategias de marketing específicas para la plataforma

📄 Análisis generado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, result)
    
    def search_game_by_name(self):
        """Buscar juego por nombre y autocompletar campos"""
        try:
            if not hasattr(self, 'original_data') or self.original_data is None:
                messagebox.showerror("Error", "Datos originales no cargados. Ejecutar analisis completo primero.")
                return
            
            search_name = self.game_vars['name_search'].get().strip().lower()
            if not search_name:
                messagebox.showwarning("Advertencia", "Por favor ingrese el nombre de un juego")
                return
            
            # Buscar juego en los datos
            matches = self.original_data[
                self.original_data['name_clean'].str.lower().str.contains(search_name, na=False)
            ]
            
            if matches.empty:
                messagebox.showinfo("No encontrado", 
                                  f"No se encontró ningún juego que contenga '{search_name}'.\n"
                                  "Puede ingresar las caracteristicas manualmente.")
                return
            
            if len(matches) > 1:
                self.show_game_selection_window(matches, search_name)
            else:
                self.populate_fields_from_game(matches.iloc[0])
                
        except Exception as e:
            messagebox.showerror("Error", f"Error buscando juego: {str(e)}")
    
    def show_game_selection_window(self, matches, search_name):
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Seleccionar Juego")
        selection_window.geometry("600x400")
        selection_window.configure(bg='#2c3e50')
        
        # titulo
        title_label = tk.Label(selection_window, 
                              text=f"Se encontraron {len(matches)} juegos que contienen '{search_name}'",
                              bg='#2c3e50', fg='white', font=('Helvetica', 12, 'bold'))
        title_label.pack(pady=10)
        
        # frame para lista
        list_frame = tk.Frame(selection_window, bg='#2c3e50')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
         # scrollbar y listbox
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        game_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                 bg='#ecf0f1', font=('Courier', 10))
        game_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=game_listbox.yview)
        
        for idx, (_, game) in enumerate(matches.iterrows()):
            game_info = f"{game['name_clean']} | {game['Genre']} | {game['platform_clean']} | {game['year_clean']}"
            game_listbox.insert(tk.END, game_info)
        
        # botones
        button_frame = tk.Frame(selection_window, bg='#2c3e50')
        button_frame.pack(pady=10)
        
        def select_game():
            selection = game_listbox.curselection()
            if selection:
                selected_game = matches.iloc[selection[0]]
                self.populate_fields_from_game(selected_game)
                selection_window.destroy()
            else:
                messagebox.showwarning("Advertencia", "Por favor seleccione un juego.")
        
        select_btn = ttk.Button(button_frame, text="Seleccionar", command=select_game)
        select_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = ttk.Button(button_frame, text="Cancelar", command=selection_window.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
    def populate_fields_from_game(self, game_data):
        try:
            # limpiar campos primero
            self.clear_fields()
            
            # llenar campos con datos reales
            self.game_vars['genre'].set(str(game_data['Genre']))
            self.game_vars['platform'].set(str(game_data['platform_clean']))
            
            # llenar scores si estan disponibles
            if pd.notna(game_data.get('Critic_Score')):
                self.game_vars['critic_score'].delete(0, tk.END)
                self.game_vars['critic_score'].insert(0, str(int(game_data['Critic_Score'])))
            
            if pd.notna(game_data.get('User_Score')):
                # convertir User_Score a numerico si es posible
                try:
                    user_score = float(game_data['User_Score'])
                    self.game_vars['user_score'].delete(0, tk.END)
                    self.game_vars['user_score'].insert(0, str(user_score))
                except:
                    pass
            
            if pd.notna(game_data.get('year_clean')):
                self.game_vars['year'].delete(0, tk.END)
                self.game_vars['year'].insert(0, str(int(game_data['year_clean'])))
            
            # ejecutar prediccion automaticamente
            self.predict_game_with_explanation(game_data)
            
            messagebox.showinfo("Éxito", f"Datos cargados para: {game_data['name_clean']}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error llenando campos: {str(e)}")
    
    def clear_fields(self):
        self.game_vars['name_search'].delete(0, tk.END)
        self.game_vars['genre'].set('')
        self.game_vars['platform'].set('')
        self.game_vars['critic_score'].delete(0, tk.END)
        self.game_vars['user_score'].delete(0, tk.END)
        self.game_vars['year'].delete(0, tk.END)
        self.result_text.delete(1.0, tk.END)
    
    def predict_game_with_explanation(self, game_data=None):
        try:
            if self.model is None:
                messagebox.showerror("Error", "Modelo no cargado. Ejecute el análisis completo primero.")
                return
            
            # obtener datos del juego
            if game_data is not None:
                # usar datos reales del juego
                genre = str(game_data['Genre'])
                platform = str(game_data['platform_clean'])
                critic_score = float(game_data.get('Critic_Score', 75))
                user_score = float(game_data.get('User_Score', 7.5)) if pd.notna(game_data.get('User_Score')) else 7.5
                year = int(game_data.get('year_clean', 2020))
                game_name = str(game_data['name_clean'])
                real_sales = float(game_data.get('Global_Sales', 0))
            else:
                # usar datos ingresados manualmente
                if not all([self.game_vars['genre'].get(), self.game_vars['platform'].get()]):
                    messagebox.showerror("Error", "Por favor complete al menos Género y Plataforma.")
                    return
                
                genre = self.game_vars['genre'].get()
                platform = self.game_vars['platform'].get()
                critic_score = float(self.game_vars['critic_score'].get() or 75)
                user_score = float(self.game_vars['user_score'].get() or 7.5)
                year = int(self.game_vars['year'].get() or 2020)
                game_name = "Juego Personalizado"
                real_sales = None
            
            # calcular probabilidad
            if self.genre_data is not None and genre in self.genre_data.index:
                base_prob = self.genre_data.loc[genre, 'Prob_Promedio']
                genre_stats = self.genre_data.loc[genre]
            else:
                base_prob = 0.5
                genre_stats = None
            
            score_factor = (critic_score / 100 + user_score / 10) / 2
            
            platform_factor = 1.0
            if self.platform_data is not None and platform in self.platform_data.index:
                platform_factor = self.platform_data.loc[platform, 'Prob_Promedio'] / 0.5
            
            # factor temporal (juegos mas recientes tienen ligera ventaja)
            year_factor = min(1.1, (year - 1980) / (2024 - 1980) * 0.2 + 0.9)
            
            final_prob = min(base_prob * score_factor * platform_factor * year_factor, 1.0)
            
            # mostrar resultado detallado
            self.show_detailed_prediction_result(
                game_name, genre, platform, critic_score, user_score, year,
                final_prob, base_prob, score_factor, platform_factor, year_factor,
                genre_stats, real_sales
            )
            
        except ValueError as e:
            messagebox.showerror("Error", "Por favor ingrese valores numéricos válidos.")
        except Exception as e:
            messagebox.showerror("Error", f"Error en predicción: {str(e)}")
    
    def show_detailed_prediction_result(self, game_name, genre, platform, critic_score, 
                                      user_score, year, final_prob, base_prob, score_factor,
                                      platform_factor, year_factor, genre_stats, real_sales):
        
        # determinar clasificacion
        if final_prob >= 0.7:
            classification = "🟢 TOP-SELLER POTENCIAL"
            confidence = "ALTA"
            recommendation = "INVERTIR - Excelentes perspectivas de éxito"
        elif final_prob >= 0.5:
            classification = "🟡 VENTAS MODERADAS"
            confidence = "MEDIA"
            recommendation = "CONSIDERAR - Buenas posibilidades con optimizaciones"
        else:
            classification = "🔴 RIESGO ELEVADO"
            confidence = "BAJA"
            recommendation = "PRECAUCIÓN - Revisar estrategia de desarrollo/marketing"
        
        result = f"""
🎮 ANÁLISIS DETALLADO DE PREDICCION
{'='*60}

📋 INFORMACIÓN DEL JUEGO:
   • Nombre: {game_name}
   • Género: {genre}
   • Plataforma: {platform}
   • Año: {year}
   • Puntuación Crítica: {critic_score}/100
   • Puntuación Usuario: {user_score}/10
"""
        
        if real_sales is not None:
            result += f"   • Ventas Reales: ${real_sales:.2f}M\n"
        
        result += f"""
🔮 RESULTADO DE LA PREDICCIÓN:
   • Probabilidad de Éxito: {final_prob:.1%}
   • Clasificación: {classification}
   • Confianza: {confidence}

💡 ANÁLISIS DE FACTORES:

📊 Factor Base del Género ({genre}): {base_prob:.1%}
"""
        
        if genre_stats is not None:
            competencia = "Alta" if genre_stats['Total_Juegos'] > 200 else "Media" if genre_stats['Total_Juegos'] > 50 else "Baja"
            result += f"""   • Competencia en el género: {competencia} ({genre_stats['Total_Juegos']} juegos)
   • Ventas promedio del género: ${genre_stats['Ventas_Promedio']:.2f}M
   • Tasa de éxito histórica: {genre_stats['Tasa_Exito_Real']:.1%}
"""
        
        result += f"""
🎯 Factor de Puntuaciones: {score_factor:.1%}
   • Impacto críticas: {(critic_score/100):.1%}
   • Impacto usuarios: {(user_score/10):.1%}
   • {"✅ Puntuaciones sólidas" if score_factor >= 0.7 else "⚠️ Puntuaciones mejorables" if score_factor >= 0.5 else "❌ Puntuaciones bajas"}

🎮 Factor de Plataforma ({platform}): {platform_factor:.1%}
"""
        
        if self.platform_data is not None and platform in self.platform_data.index:
            platform_stats = self.platform_data.loc[platform]
            result += f"""   • Probabilidad promedio en {platform}: {platform_stats['Prob_Promedio']:.1%}
   • Juegos analizados en la plataforma: {platform_stats['Total_Juegos']}
   • {"✅ Plataforma favorable" if platform_factor >= 1.0 else "⚠️ Plataforma menos favorable"}
"""
        
        result += f"""
📅 Factor Temporal: {year_factor:.1%}
   • {"✅ Timing favorable" if year_factor >= 1.0 else "⚠️ Timing neutro"}

🎯 RECOMENDACIÓN ESTRATÉGICA:
   {recommendation}

📈 ACCIONES SUGERIDAS:
"""
        
        if final_prob >= 0.7:
            result += """   ✅ Proceder con el desarrollo/lanzamiento
   ✅ Invertir en marketing premium
   ✅ Considerar expansiones post-lanzamiento
   ✅ Aprovechar el momentum del género"""
        elif final_prob >= 0.5:
            result += """   🔄 Optimizar puntuaciones de crítica/usuario
   🔄 Evaluar timing de lanzamiento
   🔄 Considerar cambios en características
   🔄 Marketing focalizegameo"""
        else:
            result += """   ⚠️ Revisar concepto fundamental del juego
   ⚠️ Considerar cambio de género o plataforma
   ⚠️ Investigar mercado objetivo
   ⚠️ Reducir inversión de marketing"""
        
        if genre_stats is not None:
            if genre_stats['Total_Juegos'] > 200:
                result += f"\n   📊 NOTA: El género {genre} está saturado - considerar diferenciación"
            
            if genre_stats['Prob_Promedio'] < 0.4:
                result += f"\n   ⚠️ ALERTA: {genre} es un género de bajo rendimiento histórico"
        
        result += f"""

🔍 COMPARACIÓN CON EL MERCADO:
   • Probabilidad vs promedio del género: {final_prob - base_prob:+.1%}
   • {"🏆 Por encima del promedio" if final_prob > base_prob else "📉 Por debajo del promedio"}

🕒 Análisis generado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, result)
    
    def run_complete_analysis(self):
        try:
            self.update_status("Ejecutando análisis completo... ⏳")
            self.root.update()
            
            # ejecutar análisis
            genre_indicators, platform_indicators, insights, results_df = self.predictor.run_complete_analysis()
            
            # actualizar datos
            self.genre_data = genre_indicators
            self.platform_data = platform_indicators
            
            # mostrar resultados
            self.show_analysis_results(genre_indicators, platform_indicators, insights)
            
            # actualizar gráficos
            self.update_charts(genre_indicators, platform_indicators)
            
            self.update_status("Análisis completado ✅")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error ejecutando análisis: {str(e)}")
            self.update_status("Error en análisis ❌")
    
    def show_analysis_results(self, genre_indicators, platform_indicators, insights):
        result = f"""
🎮 REPORTE DE ANÁLISIS COMPLETO
{'='*80}

📊 GÉNEROS DE ALTO POTENCIAL:
"""
        high_potential = genre_indicators[genre_indicators['Categoria_Riesgo'] == 'Alto Potencial']
        for genre in high_potential.index:
            prob = high_potential.loc[genre, 'Prob_Promedio']
            games = high_potential.loc[genre, 'Total_Juegos']
            result += f"   • {genre}: {prob:.1%} probabilidad ({games} juegos)\n"
        
        result += f"""
🎮 TOP 5 PLATAFORMAS:
"""
        for i, (platform, row) in enumerate(platform_indicators.head(5).iterrows(), 1):
            result += f"   {i}. {platform}: {row['Prob_Promedio']:.1%} ({row['Total_Juegos']} juegos)\n"
        
        result += f"""
💡 COMBINACIONES EXITOSAS (Género-Plataforma):
"""
        for combo in insights['combinaciones_exitosas'][:5]:
            result += f"   • {combo['Genre']} en {combo['Platform']}: {combo['Success_Probability']:.1%}\n"
        
        result += f"""
📈 RECOMENDACIONES DE INVERSIÓN:
"""
        for rec in insights['recomendaciones_inversion'][:3]:
            result += f"""
   🔸 {rec['genero']}:
      - Probabilidad: {rec['probabilidad_exito']:.1%}
      - Competencia: {rec['competencia']}
      - Ventas esperadas: ${rec['ventas_promedio']:.2f}M
"""
        
        self.analysis_text.delete(1.0, tk.END)
        self.analysis_text.insert(1.0, result)
    
    def update_charts(self, genre_indicators, platform_indicators):
        # limpiar frame anterior
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # crear figura
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Indicadores Predictivos de Éxito', fontsize=14, fontweight='bold')
        
        # grafico 1: top generos
        top_genres = genre_indicators['Prob_Promedio'].sort_values(ascending=False).head(8)
        colors = ['#2ecc71' if x >= 0.7 else '#f39c12' if x >= 0.5 else '#e74c3c' for x in top_genres.values]
        ax1.bar(range(len(top_genres)), top_genres.values, color=colors)
        ax1.set_title('Probabilidad de Éxito por Género')
        ax1.set_xlabel('Géneros')
        ax1.set_ylabel('Probabilidad')
        ax1.set_xticks(range(len(top_genres)))
        ax1.set_xticklabels(top_genres.index, rotation=45, ha='right')
        
        # grafico 2: top plataformas
        top_platforms = platform_indicators['Prob_Promedio'].head(8)
        ax2.bar(range(len(top_platforms)), top_platforms.values, color='lightblue')
        ax2.set_title('Top Plataformas por Probabilidad')
        ax2.set_xlabel('Plataformas')
        ax2.set_ylabel('Probabilidad')
        ax2.set_xticks(range(len(top_platforms)))
        ax2.set_xticklabels(top_platforms.index, rotation=45, ha='right')
        
        # grafico 3: distribucion de juegos por genero
        genre_counts = genre_indicators['Total_Juegos'].sort_values(ascending=False).head(8)
        ax3.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%')
        ax3.set_title('Distribución de Juegos por Género')
        
        # grafico 4: ventas promedio vs probabilidad
        ax4.scatter(genre_indicators['Ventas_Promedio'], genre_indicators['Prob_Promedio'], 
                   s=genre_indicators['Total_Juegos']*2, alpha=0.6, c=range(len(genre_indicators)))
        ax4.set_xlabel('Ventas Promedio (M)')
        ax4.set_ylabel('Probabilidad de Éxito')
        ax4.set_title('Ventas vs Probabilidad de Éxito')
        
        plt.tight_layout()
        
        # integrar en tkinter
        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_indicators(self):
        if self.genre_data is None:
            messagebox.showinfo("Info", "Ejecute el análisis completo primero.")
            return
        
        # cambiar a pestanna de indicadores
        self.notebook.select(2)
        
        # actualizar graficos
        self.update_charts(self.genre_data, self.platform_data)
    
    def export_report(self):
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                content = self.analysis_text.get(1.0, tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                messagebox.showinfo("Éxito", f"Reporte exportado a: {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error exportando reporte: {str(e)}")
    
    def update_status(self, message):
        if hasattr(self, 'status_label'):
            self.status_label.config(text=message)
        print(message)  

def main():
    root = tk.Tk()
    app = GamePredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 