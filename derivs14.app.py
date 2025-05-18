import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Calculadora de Volatilidade e Opções", layout="centered")

st.title("Calculadora de Volatilidade e Opções")

# Sidebar: parâmetros do ativo
st.sidebar.header("Dados do Ativo")
ticker = st.sidebar.text_input("Ticker (ex: AAPL)", value="AAPL")
periodo = st.sidebar.selectbox("Período Histórico", ["1mo", "3mo", "6mo", "1y"], index=3)

if st.sidebar.button("Carregar Dados"):
    S0, mu, sigma_hist, dados = capturar_parametros(ticker, periodo)
    st.sidebar.write(f"Preço Atual (S0): {S0:.2f}")
    st.sidebar.write(f"Retorno Médio (μ): {mu:.2%}")
    st.sidebar.write(f"Vol. Histórica (σ): {sigma_hist:.2%}")
    st.line_chart(dados['Close'], use_container_width=True)

# Funções de cálculo
@st.cache

def capturar_parametros(ticker: str, periodo: str):
    dados = yf.download(ticker, period=periodo)
    dados['Return'] = dados['Close'].pct_change()
    dados = dados.dropna()
    S0 = dados['Close'].iloc[-1]
    mu = dados['Return'].mean()
    sigma_hist = dados['Return'].std()
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
        payoffs.append(max(path.mean()-K,0))
    return np.exp(-r*T) * np.mean(payoffs)


def binomial_americana_call(S0, K, T, r, sigma, n_steps):
    dt = T/n_steps
    up, down = np.exp(sigma*np.sqrt(dt)), np.exp(-sigma*np.sqrt(dt))
    p = (np.exp(r*dt) - down)/(up-down)
    asset = np.array([S0*(up**j)*(down**(n_steps-j)) for j in range(n_steps+1)])
    vals = np.maximum(asset-K,0)
    for i in range(n_steps-1,-1,-1):
        asset = asset[:i+1]/down
        vals = np.exp(-r*dt)*(p*vals[1:i+2]+(1-p)*vals[:i+1])
        vals = np.maximum(vals, asset-K)
    return vals[0]


def plot_trajetorias(S0, T, r, sigma, n_sim, m=50):
    dt = T/m
    paths = np.zeros((m+1, n_sim))
    paths[0] = S0
    for t in range(1, m+1):
        Z = np.random.standard_normal(n_sim)
        paths[t] = paths[t-1]*np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    fig, ax = plt.subplots()
    ax.plot(paths[:,:min(10,n_sim)])
    ax.set_title('Trajetórias Monte Carlo')
    return fig

# Interface de precificação
col1, col2 = st.columns(2)
with col1:
    opt = st.selectbox("Tipo de Opção", ["Europeia","Asiática","Americana"])
    S0 = st.number_input("Preço Atual (S0)",100.0)
    K = st.number_input("Strike",S0*1.05)
    if opt=="Americana": n_steps = st.number_input("Passos Binomial",100)
with col2:
    T = st.number_input("Tempo (anos)",1.0)
    r = st.number_input("r",0.04)
    sigma = st.number_input("σ",0.2)

n_sim = st.slider("Simulações",1000,50000,20000)
if st.button("Calcular"):
    if opt=="Europeia": price=monte_carlo_opcao_europeia(S0,K,T,r,sigma,n_sim)
    elif opt=="Asiática": price=monte_carlo_opcao_asiatica(S0,K,T,r,sigma,n_sim)
    else: price=binomial_americana_call(S0,K,T,r,sigma,n_steps)
    st.write(f"Preço: {price:.4f}")
    st.pyplot(plot_trajetorias(S0,T,r,sigma,n_sim))
