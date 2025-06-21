# demo_search_games.py
"""
Script de demostración para probar la funcionalidad de búsqueda de juegos
en la interfaz gráfica
"""

import pandas as pd
import os

def demo_game_search():
    """Demostrar funcionalidad de búsqueda de juegos"""
    print("🎮 DEMOSTRACIÓN DE BÚSQUEDA DE JUEGOS")
    print("=" * 50)
    
    # Cargar datos si existen
    if not os.path.exists('Data/Processed/vgsales_integrated_refined.csv'):
        print("❌ Datos no encontrados. Ejecute primero:")
        print("python run_system.py --prepare")
        return
    
    # Cargar datos
    df = pd.read_csv('Data/Processed/vgsales_integrated_refined.csv')
    print(f"📊 Datos cargados: {len(df)} juegos disponibles")
    
    # Mostrar algunos ejemplos de juegos para buscar
    print("\n🔍 EJEMPLOS DE JUEGOS PARA BUSCAR:")
    
    # Juegos populares por género
    popular_games = [
        "super mario",
        "call of duty",
        "fifa",
        "pokemon",
        "grand theft auto",
        "zelda",
        "halo",
        "metal gear"
    ]
    
    for search_term in popular_games:
        matches = df[df['name_clean'].str.lower().str.contains(search_term, na=False)]
        if not matches.empty:
            top_match = matches.iloc[0]
            print(f"\n🎯 Buscar: '{search_term}'")
            print(f"   Encontrado: {top_match['name_clean']}")
            print(f"   Género: {top_match['Genre']} | Plataforma: {top_match['platform_clean']}")
            print(f"   Año: {top_match['year_clean']} | Ventas: ${top_match['Global_Sales']:.2f}M")
            
            # Mostrar críticas si están disponibles
            if pd.notna(top_match.get('Critic_Score')):
                print(f"   Crítica: {top_match['Critic_Score']}/100")
            if pd.notna(top_match.get('User_Score')):
                print(f"   Usuario: {top_match['User_Score']}/10")
    
    print("\n💡 INSTRUCCIONES DE USO:")
    print("1. Ejecute: python run_system.py --gui")
    print("2. En la pestaña 'Predicción Individual':")
    print("3. Escriba parte del nombre del juego en 'Buscar Juego por Nombre'")
    print("4. Haga clic en '🔍 Buscar'")
    print("5. Seleccione el juego de la lista si hay múltiples resultados")
    print("6. El sistema auto-completará los campos y mostrará análisis detallado")
    
    print("\n🎯 EJEMPLOS DE BÚSQUEDAS EXITOSAS:")
    print("- 'mario' → Super Mario Bros, Super Mario World, etc.")
    print("- 'cod' → Call of Duty series")
    print("- 'fifa' → FIFA series")
    print("- 'pokemon' → Pokemon series")
    print("- 'zelda' → The Legend of Zelda series")
    
    # Estadísticas por género
    print("\n📊 DISTRIBUCIÓN POR GÉNERO:")
    genre_counts = df['Genre'].value_counts()
    for genre, count in genre_counts.head(10).items():
        print(f"   {genre}: {count} juegos")
    
    print(f"\n✅ Sistema listo para búsquedas de {len(df)} juegos!")

if __name__ == "__main__":
    demo_game_search() 