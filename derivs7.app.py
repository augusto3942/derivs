import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# --- Tema Customizado via CSS ---
st.markdown("""
<style>
.block-container { background-color: #f0f4f8 !important; }
[data-testid="stSidebar"] { background-color: #e1ecf7 !important; }
.stButton>button { background-color: #1f77b4 !important; color: white !important; }
h1, h2 { color: #03045e !important; }
</style>
""", unsafe_allow_html=True)

# Sugestões de tickers para autocomplete
TICKER_SUGGESTIONS = ['AAPL','MSFT','GOOGL','AMZN','TSLA','KO','BRK-B','V','META','PETR4.SA','VALE3.SA','ITUB4.SA']

# --- Funções de cálculo ---
def capturar_parametros(ticker: str, periodo: str) -> tuple:
    dados = yf.download(ticker, period=periodo)
    dados['Return'] = dados['Close'].pct_change()
    dados = dados.dropna()
    return float(dados['Close'].iloc[-1]), float(dados['Return'].mean()), float(dados['Return'].std()), dados

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
    up = np.exp(sigma*np.sqrt(dt))
    down = 1 / up
    p = (np.exp(r*dt) - down) / (up - down)
    asset = np.array([S0 * (up**j) * (down**(n_steps-j)) for j in range(n_steps+1)])
    vals = np.maximum(asset - K, 0)
    for i in range(n_steps-1, -1, -1):
        asset = asset[:i+1] / down
        vals = np.exp(-r*dt) * (p * vals[1:i+2] + (1-p) * vals[:i+1])
        vals = np.maximum(vals, asset - K)
    return vals[0]

def plot_trajetorias(S0, T, r, sigma, n_sim, m=50):
    dt = T / m
    paths = np.zeros((m+1, n_sim))
    paths[0] = S0
    for t in range(1, m+1):
        Z = np.random.standard_normal(n_sim)
        paths[t] = paths[t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    fig, ax = plt.subplots()
    ax.plot(paths[:, :min(10, n_sim)], alpha=0.6)
    ax.set_xlabel('Passo')
    ax.set_ylabel('Preço')
    ax.set_title('Trajetórias MC')
    return fig

# --- Layout Streamlit ---
st.set_page_config(page_title="Calculadora de Volatilidade e Opções", layout="centered")
st.title("Calculadora de Volatilidade e Opções")

# Sidebar: seleção de ativo
st.sidebar.header("Dados do Ativo")
ticker = st.sidebar.selectbox("Ticker", TICKER_SUGGESTIONS, help="Comece a digitar para buscar")
periodo = st.sidebar.selectbox("Período Histórico", ["1mo","3mo","6mo","1y"], index=3)
if st.sidebar.button("Carregar Dados"):
    try:
        S0, mu, sigma_hist, dados = capturar_parametros(ticker, periodo)
        st.sidebar.write(f"Preço (S0): {S0:.2f}")
        st.sidebar.write(f"Vol. Histórica: {sigma_hist:.2%}")
        st.line_chart(dados['Close'], use_container_width=True)
    except Exception as e:
        st.sidebar.error(f"Erro: {e}")

# Parâmetros de precificação
col1, col2 = st.columns(2)
with col1:
    style = st.selectbox("Estilo da Opção", ["Europeia","Asiática","Americana"])
    S0_input = st.number_input("S0", value=S0 if 'S0' in locals() else 100.0)
    K = st.number_input("Strike (K)", value=S0_input * 1.05)
    if style == "Americana":
        n_steps = st.number_input("Passos Binomial", min_value=10, max_value=1000, value=100, step=10)
with col2:
    T = st.number_input("Tempo (anos)", value=1.0, min_value=0.01, step=0.01)
    r = st.number_input("r (juros)", value=0.04, format="%.4f")
    sigma_input = st.number_input("σ", value=sigma_hist if 'sigma_hist' in locals() else 0.2, format="%.4f")

n_sim = st.slider("Simulações", min_value=1000, max_value=100000, value=20000, step=1000)

if st.button("Calcular"):
    try:
        if style == "Europeia":
            preco = monte_carlo_opcao_europeia(S0_input, K, T, r, sigma_input, n_sim)
        elif style == "Asiática":
            preco = monte_carlo_opcao_asiatica(S0_input, K, T, r, sigma_input, n_sim)
        else:
            preco = binomial_americana_call(S0_input, K, T, r, sigma_input, n_steps)
        st.success(f"Preço {style}: {preco:.4f}")
        st.pyplot(plot_trajetorias(S0_input, T, r, sigma_input, n_sim))
    except Exception as e:
        st.error(f"Erro: {e}")

# Volatilidade Implícita (Europa)
st.markdown("---")
st.subheader("Volatilidade Implícita (Europeia)")
mp = st.number_input("Preço de Mercado", value=0.0)
if st.button("Calcular IV"):
    try:
        lo, hi = 1e-4, 5.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if monte_carlo_opcao_europeia(S0_input, K, T, r, mid, n_sim) - mp > 0:
                hi = mid
            else:
                lo = mid
        iv = (lo + hi) / 2
        st.write(f"IV: {iv:.2%}")
    except Exception as e:
        st.error(f"Erro: {e}")
