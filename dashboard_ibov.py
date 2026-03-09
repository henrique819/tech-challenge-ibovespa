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
        # Coleta os dados
        df = yf.download("^BVSP", period="2y", interval="1d", progress=False)
        
        if df.empty:
            return None, None, None
            
        # --- SOLUÇÃO DO ERRO: Simplifica as colunas (Remove MultiIndex) ---
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        # Garante que as colunas sejam apenas os nomes necessários
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
    except Exception as e:
        return None, None, None

    # 2. Engenharia de Features
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
    
    # 3. Target
    df['Fechamento_Amanha'] = df['Close'].shift(-1)
    df['Tendencia'] = (df['Fechamento_Amanha'] > df['Close']).astype(int)
    df = df.dropna()
    
    # 4. Separação Treino/Teste
    dias_teste = 30
    treino = df.iloc[:-dias_teste]
    teste = df.iloc[-dias_teste:]
    
    features = ['Retorno', 'Retorno_Ontem', 'Retorno_Anteontem', 'Dia_Semana', 'Variacao_Volume', 'Dist_Banda_Sup']
    
    X_treino = treino[features]
    y_treino = treino['Tendencia']
    X_teste = teste[features]
    y_teste = teste['Tendencia']
    
    # 5. Normalização
    scaler = StandardScaler()
    X_treino_scaled = scaler.fit_transform(X_treino)
    X_teste_scaled = scaler.transform(X_teste)
    
    # 6. Treinamento do Modelo Vencedor
    modelo = GradientBoostingClassifier(learning_rate=0.2, max_depth=2, n_estimators=200, random_state=42)
    modelo.fit(X_treino_scaled, y_treino)
    
    previsoes = modelo.predict(X_teste_scaled)
    acc = accuracy_score(y_teste, previsoes)
    
    return teste, previsoes, acc

# Executa a função principal
teste_df, previsoes_modelo, acuracia = carregar_e_treinar_modelo()

# Verificação se os dados carregaram
if teste_df is None:
    st.error("⚠️ Erro ao carregar dados do Yahoo Finance. Tente recarregar a página em alguns instantes.")
    st.stop()

# ==========================================
# CONSTRUÇÃO DO DASHBOARD
# ==========================================

st.title("📊 Painel Preditivo Quantitativo - IBOVESPA")
st.markdown("Desenvolvido para apoiar decisões estratégicas do time de investimentos.")

st.divider()

# --- SEÇÃO 1: RESULTADOS ---
col1, col2, col3 = st.columns(3)
col1.metric("Acurácia no Teste", f"{acuracia * 100:.2f}%", "Meta: 75%")
col2.metric("Período de Dados", "2 Anos", "Histórico")
col3.metric("Modelo", "Gradient Boosting", "Tuned")

# --- SEÇÃO 2: GRÁFICO ---
st.header("Sinais de Tendência (Últimos 30 Dias)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=teste_df.index, y=teste_df['Close'], mode='lines', name='IBOV', line=dict(color='white')))

cores = ['#00FF00' if p == 1 else '#FF0000' for p in previsoes_modelo]
fig.add_trace(go.Scatter(
    x=teste_df.index, y=teste_df['Close'], mode='markers', name='Previsão',
    marker=dict(color=cores, size=10, symbol=['triangle-up' if p==1 else 'triangle-down' for p in previsoes_modelo])
))
fig.update_layout(template="plotly_dark", height=450)
st.plotly_chart(fig, use_container_width=True)

# --- SEÇÃO 3: TABELA ---
with st.expander("Ver Tabela Detalhada"):
    tabela = pd.DataFrame({
        'Preço': teste_df['Close'].values.flatten().round(2),
        'Real': ['▲ Alta' if v == 1 else '▼ Baixa' for v in teste_df['Tendencia']],
        'Previsão': ['▲ Alta' if v == 1 else '▼ Baixa' for v in previsoes_modelo]
    }, index=teste_df.index)
    
    def colorir(v):
        if 'Alta' in str(v): return 'color: #00FF00'
        if 'Baixa' in str(v): return 'color: #FF0000'
        return ''
        
    st.dataframe(tabela.style.applymap(colorir), use_container_width=True)