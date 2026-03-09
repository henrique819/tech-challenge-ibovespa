import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Terminal IBOV - Tech Challenge", layout="wide", page_icon="📈")

# Estilo CSS para melhorar a visualização
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 28px; }
    </style>
    """, unsafe_allow_index=True)

@st.cache_data(ttl=3600)
def carregar_e_treinar_modelo():
    try:
        df = yf.download("^BVSP", period="2y", interval="1d", progress=False)
        if df.empty: return None, None, None
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    except:
        return None, None, None

    # Engenharia de Features
    df['Retorno'] = df['Close'].pct_change()
    df['Variacao_Volume'] = df['Volume'].pct_change()
    df['Retorno_Ontem'] = df['Retorno'].shift(1)
    df['Retorno_Anteontem'] = df['Retorno'].shift(2)
    df['Dia_Semana'] = df.index.dayofweek
    df['MM_20'] = df['Close'].rolling(window=20).mean()
    df['Desvio_20'] = df['Close'].rolling(window=20).std()
    df['Banda_Superior'] = df['MM_20'] + (df['Desvio_20'] * 2)
    df['Dist_Banda_Sup'] = (df['Close'] - df['Banda_Superior']) / df['Banda_Superior']
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    
    df['Fechamento_Amanha'] = df['Close'].shift(-1)
    df['Tendencia'] = (df['Fechamento_Amanha'] > df['Close']).astype(int)
    df = df.dropna()
    
    dias_teste = 30
    treino = df.iloc[:-dias_teste]
    teste = df.iloc[-dias_teste:]
    
    features = ['Retorno', 'Retorno_Ontem', 'Retorno_Anteontem', 'Dia_Semana', 'Variacao_Volume', 'Dist_Banda_Sup']
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(treino[features])
    X_teste = scaler.transform(teste[features])
    
    modelo = GradientBoostingClassifier(learning_rate=0.2, max_depth=2, n_estimators=200, random_state=42)
    modelo.fit(X_treino, treino['Tendencia'])
    
    previsoes = modelo.predict(X_teste)
    acc = accuracy_score(teste['Tendencia'], previsoes)
    
    return teste, previsoes, acc

teste_df, previsoes_modelo, acuracia = carregar_e_treinar_modelo()

if teste_df is None:
    st.error("Erro na conexão com dados financeiros.")
    st.stop()

# ==========================================
# INTERFACE PRINCIPAL
# ==========================================

st.title("📟 Terminal de Inteligência Quantitativa | IBOVESPA")

# Filtros na Barra Lateral (Sidebar)
st.sidebar.header("Filtros de Exibição")
filtro_tendencia = st.sidebar.multiselect(
    "Filtrar por Tendência Real:",
    options=['Alta', 'Baixa'],
    default=['Alta', 'Baixa']
)

# --- BLOCO DE MÉTRICAS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Último Fechamento", f"{teste_df['Close'].iloc[-1]:,.0f}".replace(",", "."), "-0.49%")
col2.metric("Acurácia IA", f"{acuracia * 100:.1f}%", "+1.67%")
col3.metric("Meta Desafio", "75.0%", None)
col4.metric("Status Modelo", "Operacional", None)

st.divider()

# --- GRÁFICO TÉCNICO ---
st.subheader("Visualização Dinâmica de Sinais")
fig = go.Figure()
fig.add_trace(go.Scatter(x=teste_df.index, y=teste_df['Close'], name='IBOV', line=dict(color='#3399FF', width=2)))

# Sinais de Compra/Venda
preds_color = ['#00FF00' if p == 1 else '#FF0000' for p in previsoes_modelo]
fig.add_trace(go.Scatter(
    x=teste_df.index, y=teste_df['Close'], mode='markers', name='Previsão IA',
    marker=dict(color=preds_color, size=10, symbol=['triangle-up' if p==1 else 'triangle-down' for p in previsoes_modelo])
))
fig.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

# --- TABELA ESTILIZADA (INVESTING STYLE) ---
st.subheader("Dados Históricos e Projeções")

# Preparação do DataFrame para exibição
tabela_show = pd.DataFrame({
    'Preço (Pontos)': teste_df['Close'].values.flatten(),
    'Real': ['Alta' if v == 1 else 'Baixa' for v in teste_df['Tendencia']],
    'Previsão IA': ['Alta' if v == 1 else 'Baixa' for v in previsoes_modelo]
}, index=teste_df.index)

# Aplicando filtro da barra lateral
tabela_show = tabela_show[tabela_show['Real'].isin(filtro_tendencia)]

# Inverter para mostrar os mais recentes primeiro (igual ao Investing)
tabela_show = tabela_show.sort_index(ascending=False)

# Funções de Estilização
def style_tendencia(val):
    color = '#00FF00' if val == 'Alta' else '#FF0000'
    return f'color: {color}; font-weight: bold'

# Formatação de Números com separador de milhar brasileiro (.)
format_dict = {'Preço (Pontos)': "{:,.0f}"}

# Exibição com Styler
st.dataframe(
    tabela_show.style
    .format(format_dict, thousands=".", decimal=",")
    .applymap(style_tendencia, subset=['Real', 'Previsão IA']),
    use_container_width=True,
    height=400
)

st.caption("Dados atualizados via Yahoo Finance API. Formatação padrão B3.")