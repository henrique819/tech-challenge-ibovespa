import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Tech Challenge - IBOVESPA", layout="wide", page_icon="📈")

# ==========================================
# FUNÇÕES DE DADOS E MODELO (Em Cache)
# ==========================================
@st.cache_data(ttl=3600)
def carregar_e_treinar_modelo():
    try:
        # Coleta os dados (yf.download é mais estável no Streamlit Cloud)
        df = yf.download("^BVSP", period="2y", interval="1d", progress=False)
        
        if df.empty:
            return None, None, None
            
        # --- LIMPEZA DE COLUNAS (Resolve o erro MultiIndex/ValueError) ---
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
    except Exception:
        return None, None, None

    # 1. Engenharia de Features
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
    
    # 2. Target (Tendência do dia seguinte)
    df['Fechamento_Amanha'] = df['Close'].shift(-1)
    df['Tendencia'] = (df['Fechamento_Amanha'] > df['Close']).astype(int)
    df = df.dropna()
    
    # 3. Separação Treino/Teste (Últimos 30 dias conforme solicitado)
    dias_teste = 30
    treino = df.iloc[:-dias_teste]
    teste = df.iloc[-dias_teste:]
    
    features = ['Retorno', 'Retorno_Ontem', 'Retorno_Anteontem', 'Dia_Semana', 'Variacao_Volume', 'Dist_Banda_Sup']
    
    X_treino = treino[features]
    y_treino = treino['Tendencia']
    X_teste = teste[features]
    y_teste = teste['Tendencia']
    
    # 4. Normalização
    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled = scaler.transform(X_teste)
    
    # 5. Treinamento
    modelo = GradientBoostingClassifier(learning_rate=0.2, max_depth=2, n_estimators=200, random_state=42)
    modelo.fit(X_treino_scaled, y_treino)
    
    previsoes = modelo.predict(X_teste_scaled)
    acc = accuracy_score(y_teste, previsoes)
    
    return teste, previsoes, acc

# Execução principal
teste_df, previsoes_modelo, acuracia = carregar_e_treinar_modelo()

if teste_df is None:
    st.error("⚠️ Erro ao carregar dados do Yahoo Finance. Tente recarregar a página.")
    st.stop()

# ==========================================
# INTERFACE DO DASHBOARD
# ==========================================

st.title("📊 Painel Preditivo Quantitativo - IBOVESPA")
st.markdown("Análise de tendência baseada em Machine Learning para o Tech Challenge.")

st.divider()

# --- BLOCO 1: MÉTRICAS ---
col1, col2, col3 = st.columns(3)
col1.metric("Acurácia no Teste", f"{acuracia * 100:.2f}%", "Meta: 75%")
col2.metric("Base Histórica", "2 Anos", "Diário")
col3.metric("Janela de Teste", "30 Dias", "Último mês")

# --- BLOCO 2: GRÁFICO INTERATIVO ---
st.header("Sinais de Tendência (Últimos 30 Dias)")
fig = go.Figure()

# Linha de Preço
fig.add_trace(go.Scatter(x=teste_df.index, y=teste_df['Close'], mode='lines', name='IBOV', line=dict(color='white', width=2)))

# Marcadores coloridos
cores = ['#00FF00' if p == 1 else '#FF0000' for p in previsoes_modelo]
simbolos = ['triangle-up' if p == 1 else 'triangle-down' for p in previsoes_modelo]

fig.add_trace(go.Scatter(
    x=teste_df.index, y=teste_df['Close'], mode='markers', name='Previsão',
    marker=dict(color=cores, symbol=simbolos, size=12, line=dict(width=1, color='black')),
    hoverinfo='text',
    hovertext=['ALTA (↑)' if p == 1 else 'BAIXA (↓)' for p in previsoes_modelo]
))

fig.update_layout(template="plotly_dark", xaxis_title="Data", yaxis_title="Pontos", height=500)
st.plotly_chart(fig, use_container_width=True)

# --- BLOCO 3: TABELA FORMATADA ---
with st.expander("Ver Tabela Detalhada de Previsões"):
    # Criando DataFrame formatado
    tabela = pd.DataFrame({
        # .astype(int) remove os zeros decimais poluídos
        'Preço (Pontos)': teste_df['Close'].values.flatten().astype(int),
        'Real': ['▲ Alta' if v == 1 else '▼ Baixa' for v in teste_df['Tendencia']],
        'Previsão': ['▲ Alta' if v == 1 else '▼ Baixa' for v in previsoes_modelo]
    }, index=teste_df.index)
    
    # Limpa a data para exibir apenas o dia
    tabela.index = tabela.index.strftime('%d/%m/%Y')
    
    # Estilo colorindo o texto
    def colorir(v):
        if '▲ Alta' in str(v): return 'color: #00FF00; font-weight: bold;'
        if '▼ Baixa' in str(v): return 'color: #FF0000; font-weight: bold;'
        return ''
        
    st.dataframe(tabela.style.applymap(colorir), use_container_width=True)

st.divider()
st.info("💡 As previsões representam a direção esperada para o fechamento do dia seguinte.")