"""
Script generado a partir de models.ipynb.
Uso: importa run_model() y pásale fechas y carpeta de output.

Modelo con restricciones de cardinalidad - limita el número de activos por región.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os


def cvar_loss(w, S, alpha=0.05):
    """Calcula el Conditional Value at Risk (CVaR)"""
    portf_rets = S @ w
    var = np.percentile(portf_rets, 100 * alpha)
    cvar = var + (1 / (alpha * len(portf_rets))) * np.sum(np.maximum(var - portf_rets, 0))
    return cvar


def limite_region(peso_region):
    """Define límites de activos por región según el peso asignado"""
    if peso_region > 0.30:
        return 13
    elif peso_region > 0.10:
        return 6
    else:
        return 3


def cardinalidad_penalizacion(w, tickers_final, equities, alloc_valid, peso_min=0.01):
    """Penalización por exceso de activos por región"""
    penal = 0.0
    for _, row in alloc_valid.iterrows():
        geo = row['Geografia']
        peso_geo = row['Peso_normalizado']
        max_activos = limite_region(peso_geo)
        tickers_region = equities[equities['Geografia'] == geo]['Ticker'].tolist()
        idxs_region = [tickers_final.index(t) for t in tickers_region if t in tickers_final]
        n_activos = np.sum(w[idxs_region] >= peso_min)
        penal += max(0, n_activos - max_activos)
    return penal


def neg_sharpe_penalized_card(w, mu, Sigma, S, alloc_valid, tickers_final, equities,
                              lambda_cvar=0.2, lambda_card=3.0, alpha=0.05):
    """Función objetivo: Sharpe ratio negativo penalizado por CVaR y cardinalidad"""
    ret = np.dot(mu, w)
    vol = np.sqrt(np.dot(w, np.dot(Sigma, w)))
    sharpe = ret / vol if vol > 0 else -1e6
    cvar = cvar_loss(w, S, alpha)
    penal = cardinalidad_penalizacion(w, tickers_final, equities, alloc_valid, peso_min=0.01)
    return - (sharpe - lambda_cvar * cvar) + lambda_card * penal


def restriccion_region(idx, p):
    """Crea función de restricción para una región específica"""
    def fun(w):
        return np.sum(w[idx]) - p
    return fun


def run_model(fecha_inicio, fecha_fin, output_folder, lambda_cvar=0.2, lambda_card=3.0, 
              alpha=0.05, min_hist_years=6, umbral_peso=1e-6, excluir_periodos=None):
    """
    Ejecuta el modelo con restricciones de cardinalidad.
    
    Parámetros:
    -----------
    fecha_inicio : str
        Fecha de inicio para el entrenamiento (formato 'YYYY-MM-DD')
    fecha_fin : str  
        Fecha límite para el entrenamiento (formato 'YYYY-MM-DD')
    output_folder : str
        Carpeta donde guardar los resultados
    lambda_cvar : float
        Parámetro de penalización por CVaR (default: 0.2)
    lambda_card : float
        Parámetro de penalización por cardinalidad (default: 3.0)
    alpha : float
        Nivel de confianza para CVaR (default: 0.05)
    min_hist_years : int
        Años mínimos de historia requeridos (default: 6)
    umbral_peso : float
        Umbral mínimo para considerar un peso como válido (default: 1e-6)
    excluir_periodos : list of tuples, optional
        Lista de períodos a excluir del entrenamiento [(inicio, fin), ...]
        Ejemplo: [('2022-01-01', '2022-12-31')] para excluir 2022
    """
    
    print(f"=== Ejecutando Modelo Limited Assets ===")
    print(f"Período de entrenamiento: {fecha_inicio} a {fecha_fin}")
    print(f"Carpeta de salida: {output_folder}")
    
    # Crear carpeta de output si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Carga de datos y filtrado base
    try:
        prices = pd.read_excel('prices/Prices.xlsx', index_col=0)
        dict_activos = pd.read_excel('dict/dict_temp.xlsx')
        alloc = pd.read_excel('allocation/Allocation.xlsx')
    except FileNotFoundError as e:
        print(f"Error: No se pudo cargar el archivo {e.filename}")
        print("Asegúrate de que los archivos estén en las rutas correctas:")
        print("- prices/Prices.xlsx")
        print("- dict/dict_temp.xlsx") 
        print("- allocation/Allocation.xlsx")
        return
    
    # Filtrar datos por rango de fechas
    prices = prices.loc[fecha_inicio:fecha_fin].copy()
    
    # Excluir períodos específicos si se especifica
    if excluir_periodos:
        print(f"Excluyendo períodos del entrenamiento: {excluir_periodos}")
        for inicio_excl, fin_excl in excluir_periodos:
            prices = prices.drop(prices.loc[inicio_excl:fin_excl].index)
        print(f"Datos después de exclusiones: {len(prices)} observaciones")
    
    # Solo Equities
    equities = dict_activos[dict_activos['Asset Class'] == 'Equities']
    tickers_equities = equities['Ticker'].unique()
    tickers_validos = [t for t in tickers_equities if t in prices.columns]
    
    prices_eq = prices[tickers_validos].copy()
    
    # Calcular el mínimo de historia requerida en días
    # Por defecto se ajusta según el periodo total
    total_days = len(prices_eq)
    min_hist = min(total_days * 0.9, min_hist_years * 252)
    
    validos_hist = prices_eq.notna().sum(axis=0) >= min_hist
    prices_eq = prices_eq.loc[:, validos_hist]
    prices_eq = prices_eq.dropna(axis=0, how='any')
    
    if len(prices_eq.columns) == 0:
        raise ValueError("No hay activos con suficiente historia para el periodo especificado")
    
    tickers_final = list(prices_eq.columns)
    equities = equities[equities['Ticker'].isin(tickers_final)].reset_index(drop=True)
    assert list(prices_eq.columns) == list(equities['Ticker']), "Tickers no sincronizados"
    
    print(f"\n--- Activos finales por región (después de todos los filtros) ---")
    region_counts = equities.groupby('Geografia').size()
    print(region_counts)
    
    # Filtrar datos por rango de fechas
    prices_eq_train = prices_eq.loc[fecha_inicio:fecha_fin]
    
    # Excluir períodos específicos si se especifica
    if excluir_periodos:
        print(f"Excluyendo períodos del entrenamiento: {excluir_periodos}")
        for inicio_excl, fin_excl in excluir_periodos:
            # Crear máscara para excluir el período
            mask_excluir = (prices_eq_train.index >= inicio_excl) & (prices_eq_train.index <= fin_excl)
            prices_eq_train = prices_eq_train[~mask_excluir]
        print(f"Datos después de exclusiones: {len(prices_eq_train)} observaciones")
    
    # 2. Estadísticos SOLO con training
    retornos_train = np.log(prices_eq_train / prices_eq_train.shift(1)).dropna()
    mu_train = retornos_train.mean().values
    Sigma_train = retornos_train.cov().values
    S_train = retornos_train.values
    n = len(tickers_final)
    
    # 3. Restricciones
    bounds = [(0, 1)] * n
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]
    
    # Allocation solo en regiones válidas y normalizadas
    regiones_validas = []
    for _, row in alloc.iterrows():
        geo = row['Geografia']
        activos_geo = equities[equities['Geografia'] == geo]['Ticker']
        if len(activos_geo) > 0:
            regiones_validas.append(geo)
    
    alloc_valid = alloc[alloc['Geografia'].isin(regiones_validas)].copy()
    total = alloc_valid['Peso'].sum()
    alloc_valid['Peso_normalizado'] = alloc_valid['Peso'] / total
    
    print(f"\n--- Allocation normalizada solo en regiones presentes ---")
    print(alloc_valid[['Geografia', 'Peso', 'Peso_normalizado']])
    
    # Mostrar límites de cardinalidad por región
    print(f"\n--- Límites de cardinalidad por región ---")
    for _, row in alloc_valid.iterrows():
        geo = row['Geografia']
        peso_obj = row['Peso_normalizado']
        max_activos = limite_region(peso_obj)
        print(f"{geo}: {max_activos} activos máximo (peso: {peso_obj:.3f})")
    
    for _, row in alloc_valid.iterrows():
        geo = row['Geografia']
        peso_obj = row['Peso_normalizado']
        activos_geo = equities[equities['Geografia'] == geo]['Ticker'].tolist()
        indices = [tickers_final.index(tkr) for tkr in activos_geo if tkr in tickers_final]
        if indices:
            constraints.append({'type': 'eq', 'fun': restriccion_region(indices, peso_obj)})
    
    # 4. Optimización
    w0 = np.ones(n) / n
    print(f"\n--- Iniciando optimización ---")
    res = minimize(
        neg_sharpe_penalized_card, w0,
        args=(mu_train, Sigma_train, S_train, alloc_valid, tickers_final, equities, 
              lambda_cvar, lambda_card, alpha),
        bounds=bounds, constraints=constraints,
        options={'maxiter': 1000}
    )
    
    if not res.success:
        print(f"Advertencia: La optimización no convergió. Mensaje: {res.message}")
    
    # 5. Resultados
    pesos_opt = res.x
    resultado = pd.DataFrame({
        'Ticker': tickers_final,
        'Peso óptimo': pesos_opt
    })
    
    print(f"\n--- Pesos óptimos por activo (usando datos hasta {fecha_fin}) ---")
    print(resultado)
    
    # 6. Chequeo: Suma por región y comparación con Allocation.xlsx
    df_merge = pd.merge(resultado, equities[['Ticker', 'Geografia']], on='Ticker', how='left')
    pesos_por_region = df_merge.groupby('Geografia')['Peso óptimo'].sum().reset_index()
    
    comparacion = pd.merge(
        pesos_por_region,
        alloc[['Geografia', 'Peso']].rename(columns={'Peso': 'Peso_Restriccion'}),
        on='Geografia',
        how='outer'
    )
    
    print(f"\n--- Suma de pesos por región y comparación con restricciones ---")
    print(comparacion)
    
    # Verificar cardinalidad final
    print(f"\n--- Verificación de cardinalidad final ---")
    for _, row in alloc_valid.iterrows():
        geo = row['Geografia']
        peso_geo = row['Peso_normalizado']
        max_activos = limite_region(peso_geo)
        tickers_region = equities[equities['Geografia'] == geo]['Ticker'].tolist()
        idxs_region = [tickers_final.index(t) for t in tickers_region if t in tickers_final]
        n_activos_final = np.sum(pesos_opt[idxs_region] >= 0.01)
        print(f"{geo}: {n_activos_final} activos seleccionados (máximo: {max_activos})")
    
    # 7. Exportar resultados
    # Selección de activos por región
    df_merge_filtrado = df_merge[df_merge['Peso óptimo'] > umbral_peso]
    df_merge_filtrado = df_merge_filtrado.sort_values(['Geografia', 'Peso óptimo'], ascending=[True, False])
    
    output_file_region = os.path.join(output_folder, 'Seleccion_Activos_por_Region_limited.xlsx')
    with pd.ExcelWriter(output_file_region) as writer:
        for region in df_merge_filtrado['Geografia'].unique():
            df_region = df_merge_filtrado[df_merge_filtrado['Geografia'] == region][['Ticker', 'Peso óptimo']]
            df_region.to_excel(writer, sheet_name=region, index=False)
    
    # Portafolio final
    portafolio_final = resultado[resultado['Peso óptimo'] > umbral_peso].copy()
    portafolio_final['Peso óptimo'] = portafolio_final['Peso óptimo'] / portafolio_final['Peso óptimo'].sum()
    portafolio_final = portafolio_final.sort_values('Peso óptimo', ascending=False)
    
    output_file_portfolio = os.path.join(output_folder, 'Portafolio_Limited_Assets.xlsx')
    portafolio_final.rename(columns={'Peso óptimo': 'Peso'}).to_excel(output_file_portfolio, index=False)
    
    print(f"\n--- Archivos generados ---")
    print(f"- {output_file_region}")
    print(f"- {output_file_portfolio}")
    print(f"=== Modelo Limited Assets completado ===\n")
    
    return {
        'resultado': resultado,
        'portafolio_final': portafolio_final,
        'comparacion_regiones': comparacion,
        'optimization_result': res
    }


if __name__ == "__main__":
    # Ejemplo de uso
    run_model('2015-01-01', '2023-12-31', 'backtests')