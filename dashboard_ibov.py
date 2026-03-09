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
@st.cache_data(ttl=3600) # Cache de 1 hora para evitar excesso de requisições
def carregar_e_treinar_modelo():
    # 1. Coleta com Tratamento de Erro (Rate Limit)
    try:
        # yf.download é mais estável para deploys em nuvem
        df = yf.download("^BVSP", period="2y", interval="1d", progress=False)
    except:
        time.sleep(2)
        df = yf.download("^BVSP", period="2y", interval="1d", progress=False)

    if df.empty:
        return None, None, None

    # Ajuste para garantir que as colunas sejam simples (sem multi-index)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
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
    st.error("⚠️ O Yahoo Finance limitou o acesso aos dados temporariamente. Por favor, aguarde 30 segundos e atualize a página.")
    st.stop()

# ==========================================
# CONSTRUÇÃO DO DASHBOARD (STORYTELLING)
# ==========================================

st.title("📊 Painel Preditivo Quantitativo - IBOVESPA")
st.markdown("Desenvolvido para auxiliar a tomada de decisão estratégica do fundo de investimentos.")

st.divider()

# --- SEÇÃO 1: RESULTADOS ---
st.header("1. Performance do Modelo")
col1, col2, col3 = st.columns(3)

col1.metric("Acurácia no Teste (30 dias)", f"{acuracia * 100:.2f}%", "Meta: 75%", delta_color="normal")
col2.metric("Dias Analisados", "2 Anos", "Histórico")
col3.metric("Algoritmo Vencedor", "Gradient Boosting", "Tuning Aplicado")

# --- SEÇÃO 2: GRÁFICO INTERATIVO ---
st.header("2. Previsão vs Realidade (Último Mês)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=teste_df.index, y=teste_df['Close'], mode='lines', name='Preço Fechamento (IBOV)', line=dict(color='white', width=2)))

cores = ['#00FF00' if p == 1 else '#FF0000' for p in previsoes_modelo]
simbolos = ['triangle-up' if p == 1 else 'triangle-down' for p in previsoes_modelo]
textos_hover = ['Previu ALTA (↑)' if p == 1 else 'Previu BAIXA (↓)' for p in previsoes_modelo]

fig.add_trace(go.Scatter(
    x=teste_df.index, y=teste_df['Close'], mode='markers', name='Sinal do Modelo',
    marker=dict(color=cores, symbol=simbolos, size=12, line=dict(width=1, color='black')),
    hoverinfo='text', hovertext=textos_hover
))

fig.update_layout(template="plotly_dark", hovermode="x unified", height=500)
st.plotly_chart(fig, use_container_width=True)

# --- SEÇÃO 3: EXPLICAÇÃO TÉCNICA ---
st.header("3. Justificativa Técnica")

with st.expander("Por que o Gradient Boosting foi escolhido?"):
    st.write("O Gradient Boosting constrói árvores em sequência para corrigir erros anteriores, sendo ideal para padrões não-lineares da bolsa.")

with st.expander("Veja a Tabela de Previsões Bruta"):
    tabela_final = pd.DataFrame({
        'Fechamento Real': teste_df['Close'].values.flatten().round(2),
        'Movimento Real': ['▲ Alta' if val == 1 else '▼ Baixa' for val in teste_df['Tendencia']],
        'Previsão do Modelo': ['▲ Alta' if val == 1 else '▼ Baixa' for val in previsoes_modelo]
    }, index=teste_df.index)
    
    def colorir_texto(valor):
        if 'Alta' in str(valor): return 'color: #00FF00; font-weight: bold;'
        if 'Baixa' in str(valor): return 'color: #FF0000; font-weight: bold;'
        return ''
        
    st.dataframe(tabela_final.style.applymap(colorir_texto, subset=['Movimento Real', 'Previsão do Modelo']), use_container_width=True)