
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configurar el estilo de matplotlib
plt.style.use('seaborn-v0_8-darkgrid')

# Configurar la página
st.set_page_config(
	page_title="Análisis de Estrategias de Portafolio",
	page_icon="📊",
	layout="wide",
	initial_sidebar_state="expanded"
)

# Título principal
st.title("📈 Análisis de Estrategias de Portafolio")
st.markdown("---")

# Funciones auxiliares
@st.cache_data
def load_data():
	"""Cargar y procesar los datos"""
	try:
		# Cargar datos principales
		data = pd.read_excel('/Users/matiasgonzalez/Desktop/Backtesting/bbdd seleccion portafolio.xlsx', index_col=0, parse_dates=True)
		data = data.sort_index(ascending=True)
		
		# Renombrar columnas
		nombres = {
			'SPX Index': 'USA', 'MXEUG Index': 'Europa equities', 'UKX Index': 'UK', 
			'MXJP Index': 'Japon', 'MXAPJ Index': 'Asia', 'MXLA Index': 'Latam', 
			'LF98TRUU Index': 'US HY', 'LUACTRUU Index': 'US IG', 'LBEATRUH Index': 'Europa bonds', 
			'BSELTRUU Index': 'Latam corp', 'BSSUTRUU Index': 'Emerging sov', 'CABS Index': 'ABS', 
			'BCOMTR Index': 'Commodities', 'GLD US EQUITY': 'Oro', 'MXWD Index': 'World equities'
		}
		data = data.rename(columns=nombres)
		
		# Calcular retornos
		returns = data.pct_change().dropna(how="all")
		returns_modelos = returns.iloc[:, 0:14]  # Primeras 14 columnas para modelos
		
		return data, returns, returns_modelos
	except Exception as e:
		st.error(f"Error al cargar los datos: {str(e)}")
		return None, None, None

year = 10

#Benchmark Portfolio Backtest conservador 
def backtest_portafolio_bvc(returns, lookback=(365*year), rebalance_freq='QE'):
    portfolio_returns = []
    weights = None
    historical_weights = []

    rebalance_dates = returns.resample(rebalance_freq).last().index

    for date in returns.index[lookback:]:
        current_daily_returns = returns.loc[date].values
        if weights is None or date in rebalance_dates:
            weights = [0.251, 0.054, 0.011, 0.026, 0.03, 0.009, 0.023, 0.224, 0.037, 0.281, 0, 0.016, 0.01, 0.025] 
            if np.sum(weights) != 1:
                weights = weights / np.sum(weights)
            period_return = np.sum(weights * current_daily_returns)
        else:
            period_return = np.sum(weights * current_daily_returns)
            asset_value_factors = (1 + current_daily_returns)/(1+period_return)
            weights = weights * asset_value_factors
            if np.sum(weights) != 1:
                weights = weights / np.sum(weights)
            period_return = np.sum(weights * current_daily_returns)

        portfolio_returns.append(period_return)
        historical_weights.append(weights)

    return pd.Series(portfolio_returns, index=returns.index[lookback:]), pd.DataFrame(historical_weights, index=returns.index[lookback:], columns=returns.columns) 

#Benchmark Portfolio Backtest conservador 
def backtest_benchmark_conservador(returns, lookback=(365*year), rebalance_freq='QE'):
    portfolio_returns = []
    weights = None
    historical_weights = []
    activos = ['World equities', 'US IG', 'Oro']
    returns = returns[activos]
    n_assets = len(activos)

    rebalance_dates = returns.resample(rebalance_freq).last().index
    
    for date in returns.index[lookback:]:
        current_daily_returns = returns.loc[date].values
        if weights is None or date in rebalance_dates:
            weights = [.35, .6, .05] 
            if np.sum(weights) != 1:
                weights = weights / np.sum(weights)
            period_return = np.sum(weights * current_daily_returns)
        else:
            period_return = np.sum(weights * current_daily_returns)
            asset_value_factors = (1 + current_daily_returns)/(1+period_return)
            weights = weights * asset_value_factors
            if np.sum(weights) != 1:
                weights = weights / np.sum(weights)
            period_return = np.sum(weights * current_daily_returns)

        portfolio_returns.append(period_return)
        historical_weights.append(weights)

    return pd.Series(portfolio_returns, index=returns.index[lookback:]), pd.DataFrame(historical_weights, index=returns.index[lookback:], columns=returns.columns) 

#Benchmark Portfolio Backtest agresivo 
def backtest_benchmark_agresivo(returns, lookback=(365*year), rebalance_freq='QE'):
    portfolio_returns = []
    weights = None
    historical_weights = []
    activos = ['World equities', 'US IG', 'Oro']
    returns = returns[activos]
    n_assets = len(activos)

    rebalance_dates = returns.resample(rebalance_freq).last().index
    
    for date in returns.index[lookback:]:
        current_daily_returns = returns.loc[date].values
        if weights is None or date in rebalance_dates:
            weights = [.6, .35, .05] 
            if np.sum(weights) != 1:
                weights = weights / np.sum(weights)
            period_return = np.sum(weights * current_daily_returns)
        else:
            period_return = np.sum(weights * current_daily_returns)
            asset_value_factors = (1 + current_daily_returns)/(1+period_return)
            weights = weights * asset_value_factors
            if np.sum(weights) != 1:
                weights = weights / np.sum(weights)
            period_return = np.sum(weights * current_daily_returns)

        portfolio_returns.append(period_return)
        historical_weights.append(weights)

    return pd.Series(portfolio_returns, index=returns.index[lookback:]), pd.DataFrame(historical_weights, index=returns.index[lookback:], columns=returns.columns) 

#Restricciones de la cartera
def get_portfolio_asset_constraints():
        cons = [
                {'type': 'ineq', 'fun': lambda w: w[0:6].sum() - 0.25},  # At least 25% in equities
                {'type': 'ineq', 'fun': lambda w: w[6:12].sum() - 0.25},  # At least 25% in bonds
                {'type': 'ineq', 'fun': lambda w: w[12:14].sum() - 0.00},  # No lower limit for commodities
                {'type': 'ineq', 'fun': lambda w: 0.70 - w[0:6].sum()},  # Max 70% in equities
                {'type': 'ineq', 'fun': lambda w: 0.70 - w[6:12].sum()},  # Max 70% in bonds
                {'type': 'ineq', 'fun': lambda w: 0.15 - w[12:14].sum()}   # Max 15% in commodities
        ]
        return cons

def get_asset_maximums(returns_modelos):
    cons = []
    max_weight = 0.15
    for i in range(len(returns_modelos.columns)):
        cons.append({
            'type': 'ineq',
            'fun': lambda w, i=i: max_weight - w[i]
        })

    return cons

# CVaR, VaR y retorno anualizado
def cvar_loss(w, S, alpha=0.05):
    """Calcula el Conditional Value at Risk (CVaR)"""
    portf_rets = S @ w
    var = np.percentile(portf_rets, 100 * alpha)
    if var < 0:
        var = var
    else:
        var = 0
    cvar = (var - (1 / (alpha * len(portf_rets))) * np.sum(np.maximum(var - portf_rets, 0))) * np.sqrt(365)
    return -cvar

def var_loss(w, S, alpha=0.05):
    """Calcula el Value at Risk (VaR)"""
    portf_rets = S @ w
    var = np.percentile(portf_rets, 100 * alpha) * np.sqrt(365)
    if var < 0:
        var = -var
    else:
        var = 0
    return var

def calcular_retorno_anualizado(returns, days=365):
    retorno_acumulado = (1+returns).prod()
    num_años = len(returns) / days
    retorno_anualizado = (retorno_acumulado ** (1/num_años)) - 1
    return retorno_anualizado

def port_vol(w,cov):
    return np.sqrt(np.dot(w.T, np.dot(cov, w))) * np.sqrt(365)  # Annualized volatility

# Calculate the covariance matrix
def port_var(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

#Calculate the risk contribution of each asset
def calculate_risk_contribution(weights, cov_matrix):
    portfolio_variance = port_var(weights, cov_matrix)
    marginal_contrib = np.dot(cov_matrix, weights)
    risk_contrib = np.multiply(weights, marginal_contrib) / portfolio_variance
    return risk_contrib

#Crear la función objetivo que disminuye la contribución de riesgo
def risk_parity_objective(weights, cov_matrix):
    risk_contrib = calculate_risk_contribution(weights, cov_matrix)
    target_risk = np.mean(risk_contrib)
    return np.sum((risk_contrib - target_risk) ** 2)

def calculate_risk_parity_weights(returns):
    cov_matrix = returns.cov().values  # Convert to NumPy array
    n_assets = returns.shape[1]
    initial_weights = np.array([1/n_assets] * n_assets)
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        {'type': 'ineq', 'fun': lambda x: x}              # non-negative weights
    ]
    result = minimize(risk_parity_objective, 
                      initial_weights,
                      args=(cov_matrix,),
                      method='SLSQP',
                      constraints=constraints,
                      options={'ftol': 1e-12})
    return result.x

# Backtesting the strategy
def backtest_risk_parity(returns, lookback=(365*year), rebalance_freq='QE'):
    historical_weights = []
    portfolio_returns = []
    weights = None

    # Obtener las fechas de rebalanceo
    rebalance_dates = returns.resample(rebalance_freq).last().index

    for date in returns.index[lookback:]:
        current_daily_returns = returns.loc[date].values
        if weights is None or date in rebalance_dates:
            historical = returns.loc[:date].iloc[-lookback:]
            weights = calculate_risk_parity_weights(historical)
            if weights.sum() != 1:
                weights = weights / weights.sum()
        else: 
            period_return = np.sum(weights * current_daily_returns)
            asset_value_factors = (1 + current_daily_returns)/(1+period_return)
            weights = weights * asset_value_factors
            if weights.sum != 1:
                weights = weights / weights.sum()


        weights_series = pd.Series(weights, index=returns.columns)
        historical_weights.append(weights)
        period_return = np.sum(weights * returns.loc[date])
        portfolio_returns.append(period_return)

    return pd.Series(portfolio_returns, index=returns.index[lookback:]), pd.DataFrame(historical_weights, index=returns.index[lookback:], columns=returns.columns)

from datetime import timedelta
#Equal weight strategy
def backtest_equal_weigth(returns, lookback=(365*year), rebalance_freq='QE'):
    portfolio_returns = []
    weights = None
    n_assets = returns.shape[1]
    historical_weights = []

    rebalance_dates = returns.resample(rebalance_freq).last().index

    for date in returns.index[lookback:]:
        current_daily_returns = returns.loc[date].values
        if weights is None or date in rebalance_dates:
            weights = np.array([1/n_assets] * n_assets)
            if weights.sum != 1:
                weights = weights / weights.sum() # Normalize weights to sum to 1
            period_return = np.sum(weights * current_daily_returns)
        else:
            period_return = np.sum(weights * current_daily_returns)
            asset_value_factors = (1 + current_daily_returns)/(1+period_return)
            weights = weights * asset_value_factors
            if weights.sum != 1:
                weights = weights / weights.sum()
            period_return = np.sum(weights * current_daily_returns)

        #print(f"Date: {date}, Weights: {weights}, Period Return: {period_return}")
        portfolio_returns.append(period_return)
        historical_weights.append(weights)

    return pd.Series(portfolio_returns, index=returns.index[lookback:]), pd.DataFrame(historical_weights, index=returns.index[lookback:], columns=returns.columns) 

def neg_sharpe_penalizado(w, mu, cov, S, lambda_cvar=0.05, alpha=0.05):
    port_return = np.dot(w, mu)
    port_vol = np.sqrt(np.dot(w, np.dot(cov, w))) * np.sqrt(365)  # Annualized volatility
    #print(f"Port Return: {port_return}, Port Volatility: {port_vol}, Weights: {w}")
    sharpe = port_return / port_vol if port_vol > 0 else 0
    #print(f"Sharpe Ratio: {sharpe}, Weights: {w}")
    sharpe = 0.00001 if sharpe < 0 else sharpe  # Evitar sharpe negativo
    cvar = cvar_loss(w, S, alpha)
    cvar = cvar if sharpe > cvar else 0
    #print(f"CVaR: {cvar}, Sharpe: {sharpe}")
    return -(sharpe - lambda_cvar * cvar)

def calculate_mvo_weights_lim_cvar_max20(returns): 
    mu = returns.apply(calcular_retorno_anualizado,axis=0)  # Annualized mean returns
    cov = returns.cov().values
    S = returns.values
    lambda_cvar = 0.2  # Penalty factor for CVaR
    alpha = 0.05  # Confidence level for CVaR

    n = len(returns.columns)
    cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda x: x},  # non-negative weights
    ]
            
    cons_assets = get_portfolio_asset_constraints()
    cons.extend(cons_assets)
    
    # Restricción para que ningún activo tenga más de un 20% de la cartera
    cons_assets_max = get_asset_maximums(returns)
    cons.extend(cons_assets_max)

    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n
    #print(f"Initial Weights: {w0}")

    res = minimize(
        neg_sharpe_penalizado, w0, args=(mu, cov, S, lambda_cvar, alpha), method='SLSQP',
        bounds=bounds, constraints=cons, options={'maxiter': 1000}
    )
    if res.success:
        return res.x
    else:
        raise ValueError("Optimization failed: " + res.message)
    
def backtest_mvo_lim_cvar_max20(returns, lookback=(365*year), rebalance_freq='QE'):
    portfolio_returns = []
    historical_weights = []
    weights = None

    # Obtener las fechas de rebalanceo
    rebalance_dates = returns.resample(rebalance_freq).last().index

    for date in returns.index[lookback:]:
        current_daily_returns = returns.loc[date].values
        if weights is None or date in rebalance_dates:
            historical = returns.loc[:date].iloc[-lookback:]
            weights = calculate_mvo_weights_lim_cvar_max20(historical)
            if weights.sum != 1:
                weights = weights / weights.sum()
        else: 
            period_return = np.sum(weights * current_daily_returns)
            asset_value_factors = (1 + current_daily_returns)/(1+period_return)
            weights = weights * asset_value_factors
            if weights.sum != 1:
                weights = weights / weights.sum() 

        weights_series = pd.Series(weights, index=returns.columns)
        historical_weights.append(weights)
        period_return = np.sum(weights_series * returns.loc[date])
        portfolio_returns.append(period_return)

    return pd.Series(portfolio_returns, index=returns.index[lookback:]), pd.DataFrame(historical_weights, index=returns.index[lookback:], columns=returns.columns) 

def max_return(w, mu):
    port_return = np.dot(w, mu)
    return -port_return

def calculate_maxreturn_weight(returns): 
    mu = returns.apply(calcular_retorno_anualizado,axis=0)  # Annualized mean returns
    S = returns.values
    alpha = 0.05  # Confidence level for CVaR

    n = len(returns.columns)
    # Definir las restricciones
    cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda x: x}  # non-negative weights
    ]
            
    cons_assets = get_portfolio_asset_constraints()
    cons.extend(cons_assets)
    
    # Restricción para que ningún activo tenga más de un 15% de la cartera
    cons_assets_max = get_asset_maximums(returns)
    cons.extend(cons_assets_max)

    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    res = minimize(
        max_return, w0, args=(mu), method='SLSQP',
        bounds=bounds, constraints=cons, options={'maxiter': 1000}
    )

    if res.success:
        return res.x
    else:
        raise ValueError("Optimization failed: " + res.message)
    
def backtest_maxreturn_weight(returns, lookback=(365*year), rebalance_freq='QE'):
    portfolio_returns = []
    historical_weights = []
    weights = None

    # Obtener las fechas de rebalanceo
    rebalance_dates = returns.resample(rebalance_freq).last().index

    for date in returns.index[lookback:]:
        current_daily_returns = returns.loc[date].values
        if weights is None or date in rebalance_dates:
            historical = returns.loc[:date].iloc[-lookback:]
            weights = calculate_maxreturn_weight(historical)
            if weights.sum() != 1:
                weights = weights / weights.sum()
        else:
            period_return = np.sum(weights * current_daily_returns)
            asset_value_factors = (1 + current_daily_returns)/(1+period_return)
            weights = weights * asset_value_factors
            if weights.sum != 1:
                weights = weights / weights.sum()
            #print(f"Weights on {date}: {weights}")

        weights_series = pd.Series(weights, index=returns.columns)
        historical_weights.append(weights)
        period_return = np.sum(weights_series * returns.loc[date])
        portfolio_returns.append(period_return)

    return pd.Series(portfolio_returns, index=returns.index[lookback:]), pd.DataFrame(historical_weights, index=returns.index[lookback:], columns=returns.columns) 

def max_return(w, mu):
    port_return = np.dot(w, mu)
    return -port_return

def calculate_maxreturn_sdres8_weight(returns): 
    mu = returns.apply(calcular_retorno_anualizado,axis=0)  # Annualized mean returns
    S = returns.values
    alpha = 0.05  # Confidence level for CVaR

    n = len(returns.columns)
    # Definir las restricciones
    cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda x: x},  # non-negative weights
    ]
            
    cons_assets = get_portfolio_asset_constraints()
    cons.extend(cons_assets)
    
    # Restricción para que ningún activo tenga más de un 20% de la cartera
    cons_assets_max = get_asset_maximums(returns)
    cons.extend(cons_assets_max)
    
    #Resrtricción de VAR
    cons.append({
        'type': 'ineq',
        'fun': lambda w: (0.08*1.4) - var_loss(w, S, alpha) # VaR should be less than 5%
    })

    cons.append({
        'type': 'ineq',
        'fun': lambda w: 0.08 - port_vol(w, returns.cov().values)  
    })

    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    res = minimize(
        max_return, w0, args=(mu), method='SLSQP',
        bounds=bounds, constraints=cons, options={'maxiter': 1000}
    )
    if res.success:
        return res.x
    else:
        raise ValueError("Optimization failed: " + res.message)

def backtest_maxreturn_sdres8_weight(returns, lookback=(365*year), rebalance_freq='QE'):
    portfolio_returns = []
    historical_weights = []
    weights = None

    # Obtener las fechas de rebalanceo
    rebalance_dates = returns.resample(rebalance_freq).last().index

    for date in returns.index[lookback:]:
        current_daily_returns = returns.loc[date].values
        if weights is None or date in rebalance_dates:
            historical = returns.loc[:date].iloc[-lookback:]
            weights = calculate_maxreturn_sdres8_weight(historical)
            if weights.sum() != 1:
                weights = weights / weights.sum()
        else:
            period_return = np.sum(weights * current_daily_returns)
            asset_value_factors = (1 + current_daily_returns)/(1+period_return)
            weights = weights * asset_value_factors
            if weights.sum != 1:
                weights = weights / weights.sum()
            #print(f"Weights on {date}: {weights}")

        weights_series = pd.Series(weights, index=returns.columns)
        historical_weights.append(weights)
        period_return = np.sum(weights_series * returns.loc[date])
        portfolio_returns.append(period_return)

    return pd.Series(portfolio_returns, index=returns.index[lookback:]), pd.DataFrame(historical_weights, index=returns.index[lookback:], columns=returns.columns) 

def max_return(w, mu):
    port_return = np.dot(w, mu)
    return port_return

def calculate_min_var_weight(returns): 
    mu = returns.apply(calcular_retorno_anualizado,axis=0)  # Annualized mean returns
    S = returns.values
    alpha = 0.05  # Confidence level for CVaR

    n = len(returns.columns)
    cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda x: x}  # non-negative weights
    ]
            
    # Restricción para los asset class de la cartera 
    cons_assets = get_portfolio_asset_constraints()
    cons.extend(cons_assets)

    # Restricción para que ningún activo tenga más de un 20% de la cartera
    cons_assets_max = get_asset_maximums(returns)
    cons.extend(cons_assets_max)
    
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n

    res = minimize(
        var_loss, w0, args=(S, alpha), method='SLSQP',
        bounds=bounds, constraints=cons, options={'maxiter': 1000}
    )
    if res.success:
        return res.x
    else:
        raise ValueError("Optimization failed: " + res.message)
    
def backtest_min_var_weight(returns, lookback=(365*year), rebalance_freq='QE'):
    portfolio_returns = []
    historical_weights = []
    weights = None

    # Obtener las fechas de rebalanceo
    rebalance_dates = returns.resample(rebalance_freq).last().index

    for date in returns.index[lookback:]:
        current_daily_returns = returns.loc[date].values
        if weights is None or date in rebalance_dates:
            historical = returns.loc[:date].iloc[-lookback:]
            weights = calculate_min_var_weight(historical)
            if weights.sum() != 1:
                weights = weights / weights.sum()
        else:
            period_return = np.sum(weights * current_daily_returns)
            asset_value_factors = (1 + current_daily_returns)/(1+period_return)
            weights = weights * asset_value_factors
            if weights.sum != 1:
                weights = weights / weights.sum()
            #print(f"Weights on {date}: {weights}")

        weights_series = pd.Series(weights, index=returns.columns)
        historical_weights.append(weights)
        period_return = np.sum(weights_series * returns.loc[date])
        portfolio_returns.append(period_return)

    return pd.Series(portfolio_returns, index=returns.index[lookback:]), pd.DataFrame(historical_weights, index=returns.index[lookback:], columns=returns.columns) 

# Funciones para calcular métricas
def calculate_portfolio_metrics(returns_dict):
    """Calcular todas las métricas del portafolio"""
    metrics = {}
    
    for strategy, returns in returns_dict.items():
        # Retorno anualizado
        annualized_return = (1 + returns).prod() ** (365 / len(returns)) - 1
        
        # Volatilidad anualizada
        annualized_volatility = returns.std() * np.sqrt(365)
        
        # Ratio de Sharpe
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Ratio de Sortino
        downside_returns = returns[returns < 0.03]
        downside_deviation = np.sqrt((downside_returns**2).mean()) * np.sqrt(365) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - 0.03) / downside_deviation if downside_deviation > 0 else 0
        
        # Max Drawdown
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # VaR y CVaR
        var = np.percentile(returns, 5) * np.sqrt(365)
        var = -var if var < 0 else 0
        
        var_percentile = np.percentile(returns, 5)
        var_percentile = var_percentile if var_percentile < 0 else 0
        cvar = (var_percentile - (1 / (0.05 * len(returns)))) * np.sum(np.maximum(var_percentile - returns, 0)) * np.sqrt(365)
        cvar = -cvar
        
        metrics[strategy] = {
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'VaR (5%)': var,
            'CVaR (5%)': cvar
        }
    
    return pd.DataFrame(metrics).T

def run_all_strategies(returns, returns_modelos):
    """Ejecutar todas las estrategias"""
    
    # Ejecutar estrategias
    portafolio_bvc_results, weights_bvc = backtest_portafolio_bvc(returns_modelos)
    benchmark_conservador_results, weights_benchmark_conservador = backtest_benchmark_conservador(returns)
    benchmark_agresivo_results, weights_benchmark_agresivo = backtest_benchmark_agresivo(returns)
    portfolio_ew_results, weights_ew = backtest_equal_weigth(returns_modelos)
    portfolio_rp_results, weights_rp = backtest_risk_parity(returns_modelos)
    portfolio_mvo_results_lim_cvar_max20, weights_mvo_lim_cvar_max20 = backtest_mvo_lim_cvar_max20(returns_modelos)
    portfolio_maxreturn, weights_maxreturn = backtest_maxreturn_weight(returns_modelos)
    portfolio_maxreturn_sdres8, weights_maxreturn_sdres8 = backtest_maxreturn_sdres8_weight(returns_modelos)
    portfolio_min_var, weights_min_var = backtest_min_var_weight(returns_modelos)

    
    # Diccionario de retornos
    returns_dict = {
        'Portafolio BVC': portafolio_bvc_results,
        'Benchmark Conservador': benchmark_conservador_results,
        'Benchmark Agresivo': benchmark_agresivo_results,
        'Equal Weight': portfolio_ew_results,
        'Risk Parity': portfolio_rp_results,
        'MVO Max 15%': portfolio_mvo_results_lim_cvar_max20,
        'Max Return': portfolio_maxreturn,
        'Max Return SD 8%': portfolio_maxreturn_sdres8,
        'Min Variance': portfolio_min_var
    }
    
    # Diccionario de pesos
    weights_dict = {
        'Portafolio BVC': weights_bvc,
        'Benchmark Conservador': weights_benchmark_conservador,
        'Benchmark Agresivo': weights_benchmark_agresivo,
        'Equal Weight': weights_ew,
        'Risk Parity': weights_rp,
        'MVO Max 15%': weights_mvo_lim_cvar_max20,
        'Max Return': weights_maxreturn,
        'Max Return SD 8%': weights_maxreturn_sdres8,
        'Min Variance': weights_min_var
    }
    
    return returns_dict, weights_dict

def create_return_vs_risk_plot(metrics_df, x_col, y_col, title):
    """Crear gráfico de retorno vs riesgo"""
    fig = px.scatter(
        metrics_df, 
        x=x_col, 
        y=y_col,
        color=metrics_df.index,
        title=title,
        labels={x_col: x_col, y_col: y_col},
        hover_data=['Sharpe Ratio', 'Max Drawdown']
    )
    
# Personalizar el gráfico
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend_title="Estrategia",
        font=dict(size=12),
        showlegend=True,
        height=600,
        xaxis=dict(
            tickformat=".2%" # Formato de porcentaje con 2 decimales para el eje X
        ),
        yaxis=dict(
            tickformat=".2%" # Formato de porcentaje con 2 decimales para el eje Y
        )
    )
    
    return fig

def create_sharpe_vs_risk_plot(metrics_df, x_col, y_col, title):
    """Crear gráfico de retorno vs riesgo"""
    fig = px.scatter(
        metrics_df, 
        x=x_col, 
        y=y_col,
        color=metrics_df.index,
        title=title,
        labels={x_col: x_col, y_col: y_col},
        hover_data=['Sharpe Ratio', 'Max Drawdown']
    )
    
# Personalizar el gráfico
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend_title="Estrategia",
        font=dict(size=12),
        showlegend=True,
        height=600
    )
    
    return fig

def create_weights_chart(weights_dict, strategy_name):
    """Crear gráfico de pesos por estrategia"""
    if strategy_name not in weights_dict:
        return None
    
    weights_df = weights_dict[strategy_name]
    
    # Obtener los últimos pesos
    latest_weights = weights_df.iloc[-1]
    latest_weights = latest_weights[latest_weights > 0.001]  # Filtrar pesos muy pequeños
    
    # Crear gráfico de barras
    fig = px.bar(
        x=latest_weights.index,
        y=latest_weights.values,
        title=f'Composición del Portafolio - {strategy_name}',
        labels={'x': 'Activos', 'y': 'Peso (%)'},
        color=latest_weights.values,
        color_continuous_scale='viridis'
    )
    
    fig.update_traces(
        texttemplate='%{y:.1%}',
        textposition='outside'
    )
    
    fig.update_layout(
        xaxis_title="Activos",
        yaxis_title="Peso (%)",
        showlegend=False,
        height=500,
        yaxis=dict(tickformat='.1%')
    )
    
    return fig

def create_cumulative_returns_plot(returns_dict):
    """Crear gráfico de retornos acumulados"""
    cumulative_returns = pd.DataFrame({
        strategy: (1 + returns).cumprod() 
        for strategy, returns in returns_dict.items()
    })
    
    fig = px.line(
        cumulative_returns,
        title='Retornos Acumulados por Estrategia',
        labels={'index': 'Fecha', 'value': 'Valor del Portafolio'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Valor del Portafolio (Base 100)",
        legend_title="Estrategia",
        height=600
    )
    
    return fig

def calculate_max_drawdown(returns_series):
    """Calcular el máximo drawdown de una serie de retornos"""
    cumulative_returns = (1 + returns_series).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_asset_metrics(returns):
    """Calcular métricas individuales para cada activo"""
    metrics = {}
    
    for asset in returns.columns:
        asset_returns = returns[asset].dropna()
        
        # Retorno anualizado
        annualized_return = calcular_retorno_anualizado(asset_returns)
        
        # Volatilidad anualizada
        annualized_volatility = asset_returns.std() * np.sqrt(365)
        
        # Max Drawdown
        max_drawdown = calculate_max_drawdown(asset_returns)
        
        # VaR (5%)
        var_5 = np.percentile(asset_returns, 5) * np.sqrt(365)
        var_5 = -var_5 if var_5 < 0 else 0
        
        # CVaR (5%)
        var_percentile = np.percentile(asset_returns, 5)
        var_percentile = var_percentile if var_percentile < 0 else 0
        cvar_5 = var_percentile - (1 / (0.05 * len(asset_returns))) * np.sum(np.maximum(var_percentile - asset_returns, 0))
        cvar_5 = -cvar_5 * np.sqrt(365)
        
        metrics[asset] = {
            'Retorno Anualizado': annualized_return,
            'Volatilidad Anualizada': annualized_volatility,
            'Max Drawdown': max_drawdown,
            'VaR (5%)': var_5,
            'CVaR (5%)': cvar_5
        }
    
    return pd.DataFrame(metrics).T

def create_asset_correlation_heatmap(returns):
    """Crear heatmap de correlaciones entre activos"""
    correlation_matrix = returns.corr()
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        color_continuous_midpoint=0,
        title='Matriz de Correlaciones entre Activos'
    )
    
    fig.update_layout(
        xaxis_title="Activos",
        yaxis_title="Activos",
        height=600
    )
    
    return fig

def create_asset_metrics_chart(metrics_df, metric_name):
    """Crear gráfico de barras para una métrica específica"""
    fig = px.bar(
        x=metrics_df.index,
        y=metrics_df[metric_name],
        title=f'{metric_name} por Activo',
        labels={'x': 'Activo', 'y': metric_name},
        color=metrics_df[metric_name],
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="Activo",
        yaxis_title=metric_name,
        height=500,
        xaxis={'categoryorder': 'total descending'}
    )
    
    # Formatear eje Y según el tipo de métrica
    if 'Retorno' in metric_name or 'Volatilidad' in metric_name or 'Drawdown' in metric_name or 'VaR' in metric_name or 'CVaR' in metric_name:
        fig.update_layout(yaxis=dict(tickformat='.1%'))
    
    return fig

# Interfaz principal
def main():
    # Cargar datos
    data, returns, returns_modelos = load_data()
    
    if data is None:
        st.error("No se pudieron cargar los datos. Por favor, verifica la ruta del archivo.")
        return
    
    # Sidebar para navegación
    st.sidebar.title("Navegación")
    pages = ["Resumen Ejecutivo", "Análisis de Riesgo-Retorno", "Composición de Portafolios", "Métricas Detalladas", "Análisis de Activos"]
    selected_page = st.sidebar.selectbox("Selecciona una sección:", pages)
    
    # Ejecutar estrategias
    with st.spinner("Ejecutando estrategias de backtest..."):
        returns_dict, weights_dict = run_all_strategies(returns, returns_modelos)
        metrics_df = calculate_portfolio_metrics(returns_dict)
    
    if selected_page == "Resumen Ejecutivo":
        st.header("📊 Resumen Ejecutivo")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            best_return = metrics_df['Annualized Return'].max()
            best_strategy = metrics_df['Annualized Return'].idxmax()
            st.metric(
                "Mejor Retorno Anualizado",
                f"{best_return:.2%}",
                f"{best_strategy}"
            )
        
        with col2:
            best_sharpe = metrics_df['Sharpe Ratio'].max()
            best_sharpe_strategy = metrics_df['Sharpe Ratio'].idxmax()
            st.metric(
                "Mejor Sharpe",
                f"{best_sharpe:.2f}",
                f"{best_sharpe_strategy}"
            )
        
        with col3:
            min_drawdown = metrics_df['Max Drawdown'].max()  # Menos negativo
            min_drawdown_strategy = metrics_df['Max Drawdown'].idxmax()
            st.metric(
                "Menor Drawdown",
                f"{min_drawdown:.2%}",
                f"{min_drawdown_strategy}"
            )
        
        with col4:
            min_volatility = metrics_df['Annualized Volatility'].min()
            min_vol_strategy = metrics_df['Annualized Volatility'].idxmin()
            st.metric(
                "Menor Volatilidad Anualizada",
                f"{min_volatility:.2%}",
                f"{min_vol_strategy}"
            )
        
        # Gráfico de retornos acumulados
        st.subheader("Evolución de los Portafolios")
        fig_cumulative = create_cumulative_returns_plot(returns_dict)
        st.plotly_chart(fig_cumulative, use_container_width=True)
    
    elif selected_page == "Análisis de Riesgo-Retorno":
        st.header("⚖️ Análisis de Riesgo-Retorno")
        
        # Crear gráficos de dispersión
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = create_return_vs_risk_plot(
                metrics_df, 
                'Annualized Volatility', 
                'Annualized Return',
                'Retorno vs Volatilidad'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = create_return_vs_risk_plot(
                metrics_df, 
                'Max Drawdown', 
                'Annualized Return',
                'Retorno vs Max Drawdown'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig3 = create_return_vs_risk_plot(
                metrics_df, 
                'VaR (5%)', 
                'Annualized Return',
                'Retorno vs VaR (5%)'
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col4:
            fig4 = create_return_vs_risk_plot(
                metrics_df, 
                'CVaR (5%)', 
                'Annualized Return',
                'Retorno vs CVaR (5%)'
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        # Gráfico de Sharpe vs Volatilidad
        st.subheader("Eficiencia de Riesgo")
        fig5 = create_sharpe_vs_risk_plot(
            metrics_df, 
            'Annualized Volatility', 
            'Sharpe Ratio',
            'Sharpe Ratio vs Volatilidad'
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    elif selected_page == "Composición de Portafolios":
        st.header("💰 Composición de Portafolios")
        
        # Selector de estrategia
        selected_strategy = st.selectbox(
            "Selecciona una estrategia:",
            list(weights_dict.keys())
        )
        
        # Mostrar composición
        if selected_strategy:
            fig_weights = create_weights_chart(weights_dict, selected_strategy)
            if fig_weights:
                st.plotly_chart(fig_weights, use_container_width=True)
            
            # Tabla de pesos
            st.subheader("Pesos Detallados")
            weights_df = weights_dict[selected_strategy]
            latest_weights = weights_df.iloc[-1]
            
            # Crear DataFrame para mostrar
            weights_display = pd.DataFrame({
                'Activo': latest_weights.index,
                'Peso (%)': latest_weights.values
            })
            weights_display['Peso (%)'] = weights_display['Peso (%)'].apply(lambda x: f"{x:.2%}")
            weights_display = weights_display[weights_display['Peso (%)'] != '0.00%']
            
            st.dataframe(weights_display, use_container_width=True)
            
            # Evolución temporal de pesos
            st.subheader("Evolución Temporal de Pesos")
            
            # Filtrar solo activos con peso significativo
            significant_assets = latest_weights[latest_weights > 0.01].index
            weights_evolution = weights_df[significant_assets]
            
            fig_evolution = px.line(
                weights_evolution,
                title=f'Evolución de Pesos - {selected_strategy}',
                labels={'index': 'Fecha', 'value': 'Peso (%)'},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            
            fig_evolution.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Peso (%)",
                legend_title="Activo",
                height=500,
                yaxis=dict(tickformat='.1%')
            )
            
            st.plotly_chart(fig_evolution, use_container_width=True)
    
    elif selected_page == "Métricas Detalladas":
        st.header("📋 Métricas Detalladas")
        
        # Mostrar métricas completas
        st.subheader("Todas las Métricas")
        
        # Formatear el DataFrame para mejor visualización
        metrics_formatted = metrics_df.copy()
        
        # Convertir a porcentajes las columnas apropiadas
        percentage_cols = ['Annualized Return', 'Annualized Volatility', 'Max Drawdown', 'VaR (5%)', 'CVaR (5%)']
        for col in percentage_cols:
            metrics_formatted[col] = metrics_formatted[col].apply(lambda x: f"{x:.2%}")
        
        # Redondear ratios
        ratio_cols = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
        for col in ratio_cols:
            metrics_formatted[col] = metrics_formatted[col].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(metrics_formatted, use_container_width=True)
        
        # Gráficos de barras por métrica
        st.subheader("Comparación por Métricas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Retornos anualizados
            fig_returns = px.bar(
                x=metrics_df.index,
                y=metrics_df['Annualized Return'],
                title='Retornos Anualizados por Estrategia',
                labels={'x': 'Estrategia', 'y': 'Retorno Anualizado (%)'},
                color=metrics_df['Annualized Return'],
                color_continuous_scale='viridis'
            )
            fig_returns.update_layout(yaxis=dict(tickformat='.1%'))
            st.plotly_chart(fig_returns, use_container_width=True)
        
        with col2:
            # Ratios de Sharpe
            fig_sharpe = px.bar(
                x=metrics_df.index,
                y=metrics_df['Sharpe Ratio'],
                title='Ratios de Sharpe por Estrategia',
                labels={'x': 'Estrategia', 'y': 'Ratio de Sharpe'},
                color=metrics_df['Sharpe Ratio'],
                color_continuous_scale='plasma'
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Volatilidad
            fig_vol = px.bar(
                x=metrics_df.index,
                y=metrics_df['Annualized Volatility'],
                title='Volatilidad Anualizada por Estrategia',
                labels={'x': 'Estrategia', 'y': 'Volatilidad Anualizada (%)'},
                color=metrics_df['Annualized Volatility'],
                color_continuous_scale='reds'
            )
            fig_vol.update_layout(yaxis=dict(tickformat='.1%'))
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col4:
            # Max Drawdown
            fig_dd = px.bar(
                x=metrics_df.index,
                y=metrics_df['Max Drawdown'],
                title='Max Drawdown por Estrategia',
                labels={'x': 'Estrategia', 'y': 'Max Drawdown (%)'},
                color=metrics_df['Max Drawdown'],
                color_continuous_scale='oranges'
            )
            fig_dd.update_layout(yaxis=dict(tickformat='.1%'))
            st.plotly_chart(fig_dd, use_container_width=True)
    
    elif selected_page == "Análisis de Activos":
        st.header("🏦 Análisis de Activos Individuales")
        
        # Calcular métricas de activos
        asset_metrics = calculate_asset_metrics(returns)
        
        # Mostrar tabla de métricas
        st.subheader("Métricas de Activos")
        
        # Formatear el DataFrame para mejor visualización
        asset_metrics_formatted = asset_metrics.copy()
        
        # Convertir a porcentajes las columnas apropiadas
        percentage_cols = ['Retorno Anualizado', 'Volatilidad Anualizada', 'Max Drawdown', 'VaR (5%)', 'CVaR (5%)']
        for col in percentage_cols:
            asset_metrics_formatted[col] = asset_metrics_formatted[col].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(asset_metrics_formatted, use_container_width=True)
        
        # Gráficos de métricas individuales
        st.subheader("Comparación de Métricas por Activo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Retorno Anualizado
            fig_return = create_asset_metrics_chart(asset_metrics, 'Retorno Anualizado')
            st.plotly_chart(fig_return, use_container_width=True)
            
            # Max Drawdown
            fig_drawdown = create_asset_metrics_chart(asset_metrics, 'Max Drawdown')
            st.plotly_chart(fig_drawdown, use_container_width=True)
            
            # CVaR
            fig_cvar = create_asset_metrics_chart(asset_metrics, 'CVaR (5%)')
            st.plotly_chart(fig_cvar, use_container_width=True)
        
        with col2:
            # Volatilidad Anualizada
            fig_vol = create_asset_metrics_chart(asset_metrics, 'Volatilidad Anualizada')
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # VaR
            fig_var = create_asset_metrics_chart(asset_metrics, 'VaR (5%)')
            st.plotly_chart(fig_var, use_container_width=True)
        
        # Gráfico de dispersión Retorno vs Riesgo para activos
        st.subheader("Retorno vs Riesgo por Activo")
        
        fig_scatter = px.scatter(
            asset_metrics,
            x='Volatilidad Anualizada',
            y='Retorno Anualizado',
            color=asset_metrics.index,
            title='Retorno vs Volatilidad por Activo',
            labels={'Volatilidad Anualizada': 'Volatilidad Anualizada (%)', 'Retorno Anualizado': 'Retorno Anualizado (%)'},
            hover_data=['Max Drawdown', 'VaR (5%)', 'CVaR (5%)']
        )
        
        fig_scatter.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
        fig_scatter.update_layout(
            xaxis_title="Volatilidad Anualizada (%)",
            yaxis_title="Retorno Anualizado (%)",
            legend_title="Activo",
            height=600,
            xaxis=dict(tickformat=".1%"),
            yaxis=dict(tickformat=".1%")
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Matriz de correlaciones
        st.subheader("Matriz de Correlaciones")
        
        fig_corr = create_asset_correlation_heatmap(returns)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Tabla de correlaciones
        st.subheader("Tabla de Correlaciones")
        correlation_matrix = returns.corr()
        correlation_formatted = correlation_matrix.round(3)
        st.dataframe(correlation_formatted, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("**Análisis de Estrategias de Portafolio** | Desarrollado con Streamlit")

if __name__ == "__main__":
    main()