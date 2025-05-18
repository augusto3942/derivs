import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Sugestões de tickers para autocomplete
TICKER_SUGGESTIONS = [
    'AAPL','MSFT','GOOGL','AMZN','TSLA','KO','BRK-B','V','META',
    'PETR4.SA','VALE3.SA','ITUB4.SA','BBDC4.SA','ABEV3.SA'
]

# --- Funções de Cálculo ---
def capturar_parametros(ticker: str, periodo: str) -> tuple:
    """
    Baixa históricos via yfinance e retorna (S0, mu, sigma_hist, dados).
    """
    dados = yf.download(ticker, period=periodo)
    dados['Return'] = dados['Close'].pct_change()
    dados = dados.dropna()
    S0 = float(dados['Close'].iloc[-1])
    mu = float(dados['Return'].mean())
    sigma_hist = float(dados['Return'].std())
    return S0, mu, sigma_hist, dados


def monte_carlo_opcao_europeia(S0, K, T, r, sigma, n_sim):
    Z = np.random.standard_normal(n_sim)
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r*T) * payoff.mean()


def monte_carlo_opcao_asiatica(S0, K, T, r, sigma, n_sim, m=50):
    dt = T/m
    payoffs = []
    for _ in range(n_sim):
        Z = np.random.standard_normal(m)
        path = S0 * np.exp(np.cumsum((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z))
        payoffs.append(max(path.mean() - K, 0))
    return np.exp(-r*T) * np.mean(payoffs)


def binomial_americana_call(S0, K, T, r, sigma, n_steps):
    dt = T/n_steps
    up = np.exp(sigma * np.sqrt(dt))
    down = 1/up
    p = (np.exp(r*dt) - down) / (up - down)
    # Preço do ativo no vencimento
    asset_prices = np.array([S0 * (up**j) * (down**(n_steps-j)) for j in range(n_steps+1)])
    # Payoff no vencimento
    option_values = np.maximum(asset_prices - K, 0)
    # Backward induction
    for i in range(n_steps-1, -1, -1):
        asset_prices = asset_prices[:i+1] / down
        option_values = np.exp(-r*dt) * (p * option_values[1:i+2] + (1-p) * option_values[:i+1])
        # Exercício antecipado
        option_values = np.maximum(option_values, asset_prices - K)
    return option_values[0]


def plot_trajetorias(S0, T, r, sigma, n_sim, m=50):
    dt = T/m
    paths = np.zeros((m+1, n_sim))
    paths[0] = S0
    for t in range(1, m+1):
        Z = np.random.standard_normal(n_sim)
        paths[t] = paths[t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    fig, ax = plt.subplots()
    ax.plot(paths[:, :min(10, n_sim)], alpha=0.6)
    ax.set_title('Trajetórias Monte Carlo')
    ax.set_xlabel('Passo')
    ax.set_ylabel('Preço do Ativo')
    return fig

# --- App Streamlit ---
st.set_page_config(page_title="Calculadora de Volatilidade e Opções", layout="centered")
st.title("📊 Calculadora de Volatilidade e Opções")

# Sidebar: parâmetros do ativo
st.sidebar.header("Dados do Ativo")
ticker = st.sidebar.selectbox("Ticker (ex: AAPL)", TICKER_SUGGESTIONS, index=0, help="Comece a digitar para buscar")
periodo = st.sidebar.selectbox("Período Histórico", ["1mo", "3mo", "6mo", "1y"], index=3)

if st.sidebar.button("Carregar Dados"):
    try:
        S0, mu, sigma_hist, dados = capturar_parametros(ticker, periodo)
        st.sidebar.write(f"Preço Atual (S0): R$ {S0:.2f}")
        st.sidebar.write(f"Retorno Médio (μ): {mu:.2%}")
        st.sidebar.write(f"Vol. Histórica (σ): {sigma_hist:.2%}")
        st.line_chart(dados['Close'], use_container_width=True)
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar dados: {e}")

# Parâmetros de precificação
col1, col2 = st.columns(2)
with col1:
    option_style = st.selectbox("Estilo da Opção", ["Europeia", "Asiática", "Americana"])
    S0_input = st.number_input("Preço Atual (S0)", value=S0 if 'S0' in locals() else 100.0)
    K = st.number_input("Strike (K)", value=S0_input * 1.05)
    if option_style == "Americana":
        n_steps = st.number_input("Passos Binomial", min_value=10, max_value=1000, value=100, step=10)
with col2:
    T = st.number_input("Tempo até Vencimento (anos)", value=1.0, min_value=0.01, step=0.01)
    r = st.number_input("Taxa Livre de Risco (r)", value=0.04, format="%.4f")
    sigma_input = st.number_input("Volatilidade (σ)", value=sigma_hist if 'sigma_hist' in locals() else 0.2, format="%.4f")

n_sim = st.slider("Número de Simulações", 1000, 100000, 20000, 1000)

if st.button("Calcular"):
    try:
        if option_style == "Europeia":
            preco = monte_carlo_opcao_europeia(S0_input, K, T, r, sigma_input, n_sim)
        elif option_style == "Asiática":
            preco = monte_carlo_opcao_asiatica(S0_input, K, T, r, sigma_input, n_sim)
        else:
            preco = binomial_americana_call(S0_input, K, T, r, sigma_input, n_steps)
        st.success(f"Preço da Opção {option_style}: R$ {preco:.4f}")
        fig = plot_trajetorias(S0_input, T, r, sigma_input, n_sim)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erro ao calcular: {e}")

# Volatilidade Implícita
st.markdown("---")
st.subheader("🔍 Volatilidade Implícita (Europa)")
market_price = st.number_input("Preço de Mercado da Call", value=0.0)
if st.button("Calcular IV"):
    try:
        def diff(sig):
            return monte_carlo_opcao_europeia(S0_input, K, T, r, sig, n_sim) - market_price
        low, high = 1e-4, 5.0
        for _ in range(50):
            mid = (low + high) / 2
            if diff(mid) > 0:
                high = mid
            else:
                low = mid
        iv = (low + high) / 2
        st.write(f"Volatilidade Implícita: {iv:.2%}")
    except Exception as e:
        st.error(f"Erro: {e}")
