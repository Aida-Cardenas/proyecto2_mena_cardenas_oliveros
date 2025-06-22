import os
import sys
import argparse
from datetime import datetime

def print_banner():
    banner = """
    ╔════════════════════════════════════════════════════════════════╗
    ║                🎮 SISTEMA PREDICTOR DE VIDEOJUEGOS 🎮         ║
    ║                                                                ║
    ║    Predicción de Éxito Comercial usando Redes Neuronales       ║
    ║                          Versión 1.0                           ║
    ╚════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    print("🔍 Verificando dependencias...")
    
    required_packages = [
        'torch', 'pandas', 'numpy', 'scikit-learn', 
        'matplotlib', 'seaborn', 'tkinter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'scikit-learn':
                # Verificar scikit-learn específicamente
                import sklearn
                print(f"  ✅ {package} (versión: {sklearn.__version__})")
            else:
                __import__(package)
                print(f"  ✅ {package}")
        except ImportError as e:
            missing_packages.append(package)
            print(f"  ❌ {package} - NO ENCONTRADO: {str(e)}")
    
    if missing_packages:
        print(f"\n⚠️  Faltan paquetes: {', '.join(missing_packages)}")
        print("Instale con: pip install " + " ".join(missing_packages))
        print("\nSi scikit-learn sigue fallando, pruebe:")
        print("  python -m pip install --upgrade scikit-learn")
        print("  python -c \"import sklearn; print(sklearn.__version__)\"")
        return False
    
    print("✅ Todas las dependencias están instaladas")
    return True

def prepare_data():
    print("\n📊 PREPARANDO DATOS Y CARACTERÍSTICAS")
    print("=" * 50)
    
    try:
        # Verificar si existen datos procesados
        if os.path.exists("Data/Processed/features_matrix.csv"):
            print("✅ Datos ya procesados encontrados")
            return True
        
        # Importar y ejecutar preparación de datos
        from Data.preprocess import main as preprocess_main
        from Features.build_features import build_and_save_features
        
        print("🔄 Preprocesando datos...")
        preprocess_main()
        
        print("🔄 Construyendo características...")
        build_and_save_features()
        
        print("✅ Datos preparados exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error preparando datos: {str(e)}")
        return False

def train_models():
    print("\n🤖 ENTRENANDO MODELOS DE MACHINE LEARNING")
    print("=" * 50)
    
    try:
        # Verificar si existen modelos entrenados
        if os.path.exists("Models/mlp_classifier.pth") and os.path.exists("Models/baseline_perceptron.pth"):
            print("✅ Modelos ya entrenados encontrados")
            return True
        
        # Importar y ejecutar entrenamiento
        from Models.train_pytorch import main as train_main
        
        print("🔄 Entrenando modelos...")
        train_main()
        
        print("✅ Modelos entrenados exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error entrenando modelos: {str(e)}")
        return False

def generate_indicators():
    print("\n📈 GENERANDO INDICADORES PREDICTIVOS")
    print("=" * 50)
    
    try:
        from Features.predictive_indicators import PredictiveIndicators
        
        print("🔄 Ejecutando análisis de indicadores...")
        analyzer = PredictiveIndicators()
        analyzer.run_complete_analysis()
        
        print("✅ Indicadores generados exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error generando indicadores: {str(e)}")
        return False

def run_validation():
    print("\n🔍 EJECUTANDO VALIDACIÓN DEL SISTEMA")
    print("=" * 50)
    
    try:
        from Tests.test_system_validation import SystemValidationTests
        
        print("🔄 Ejecutando suite de pruebas...")
        validator = SystemValidationTests()
        success = validator.run_all_tests()
        
        if success:
            print("✅ Sistema validado exitosamente")
        else:
            print("⚠️ Sistema parcialmente validado - revisar errores")
        
        return success
        
    except Exception as e:
        print(f"❌ Error en validación: {str(e)}")
        return False

def launch_gui():
    print("\n🖥️ LANZANDO INTERFAZ GRÁFICA")
    print("=" * 50)
    
    try:
        from GUI.game_predictor_gui import main as gui_main
        
        print("🔄 Iniciando interfaz gráfica...")
        gui_main()
        
        return True
        
    except Exception as e:
        print(f"❌ Error lanzando GUI: {str(e)}")
        return False

def create_requirements_file():
    requirements = """
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements.strip())
    
    print("📝 Archivo requirements.txt creado")

def show_usage():
    usage = """
📝 INSTRUCCIONES DE USO DEL SISTEMA

COMANDOS:
  python run_system.py --full          # Ejecutar pipeline completo
  python run_system.py --prepare       # Solo preparar datos
  python run_system.py --train         # Solo entrenar modelos
  python run_system.py --indicators    # Solo generar indicadores
  python run_system.py --validate      # Solo ejecutar validación
  python run_system.py --gui           # Solo lanzar interfaz gráfica
  python run_system.py --requirements  # Crear requirements.txt
  python run_system.py --help          # Mostrar ayuda

EJEMPLOS DE USO:
  # Configuración inicial completa
  python run_system.py --full
  
  # Solo usar la interfaz gráfica (después de configuración)
  python run_system.py --gui
  
  # Re-entrenar modelos con nuevos datos
  python run_system.py --train --indicators
  
  # Validar sistema después de cambios
  python run_system.py --validate

ESTRUCTURA DE ARCHIVOS GENERADOS:
  Data/Processed/                      # Datos procesados
  Models/                             # Modelos entrenados
  Tests/validation_report.html        # Reporte de validación
  Tests/validation_plots.png          # Gráficos de validación
  Data/Processed/reporte_indicadores_predictivos.md  # Reporte de indicadores
"""
    print(usage)

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description="Sistema Predictor de Éxito en Videojuegos")
    parser.add_argument("--full", action="store_true", help="Ejecutar pipeline completo")
    parser.add_argument("--prepare", action="store_true", help="Solo preparar datos")
    parser.add_argument("--train", action="store_true", help="Solo entrenar modelos")
    parser.add_argument("--indicators", action="store_true", help="Solo generar indicadores")
    parser.add_argument("--validate", action="store_true", help="Solo ejecutar validación")
    parser.add_argument("--gui", action="store_true", help="Solo lanzar interfaz gráfica")
    parser.add_argument("--requirements", action="store_true", help="Crear requirements.txt")
    parser.add_argument("--usage", action="store_true", help="Mostrar instrucciones de uso")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        show_usage()
        return
    
    if args.usage:
        show_usage()
        return
    
    if args.requirements:
        create_requirements_file()
        return
    
    if not check_dependencies():
        print("\n❌ Faltan dependencias. Instala los paquetes requeridos.")
        return
    
    success = True
    start_time = datetime.now()
    
    if args.full:
        print("\n🚀 EJECUTANDO PIPELINE COMPLETO")
        print("=" * 60)
        
        steps = [
            ("Preparar datos", prepare_data),
            ("Entrenar modelos", train_models),
            ("Generar indicadores", generate_indicators),
            ("Validar sistema", run_validation),
            ("Lanzar GUI", launch_gui)
        ]
        
        for step_name, step_func in steps:
            print(f"\n▶️  {step_name}...")
            if not step_func():
                success = False
                print(f"❌ Error en: {step_name}")
                break
        
    else:
        if args.prepare:
            success &= prepare_data()
        
        if args.train:
            success &= train_models()
        
        if args.indicators:
            success &= generate_indicators()
        
        if args.validate:
            success &= run_validation()
        
        if args.gui:
            success &= launch_gui()
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE EJECUCIÓN")
    print("=" * 60)
    print(f"Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duración: {duration}")
    
    if success:
        print("EJECUCIÓN COMPLETADA EXITOSAMENTE")
        print("\n📁 Archivos generados:")
        
        generated_files = [
            "Data/Processed/features_matrix.csv",
            "Data/Processed/labels.csv",
            "Models/mlp_classifier.pth",
            "Models/baseline_perceptron.pth",
            "Data/Processed/reporte_indicadores_predictivos.md",
            "Tests/validation_report.html",
            "Tests/validation_plots.png"
        ]
        
        for file_path in generated_files:
            if os.path.exists(file_path):
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path} (no generado)")
        
        print("\n💡")
        print("   1. Revisa el reporte de validación en Tests/validation_report.html")
        print("   2. Usa la interfaz gráfica: python run_system.py --gui")
        print("   3. Consulta indicadores en Data/Processed/reporte_indicadores_predictivos.md")
        
    else:
        print("❌ EJECUCIÓN CON ERRORES")
        print("   Revisa los mensajes de error anteriores")
        print("   Verifica que todos los archivos de datos estén presentes")

if __name__ == "__main__":
    main() 