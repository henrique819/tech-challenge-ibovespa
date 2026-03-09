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
st.set_page_config(page_title="Terminal IBOV - Tech Challenge", layout="wide", page_icon="📈")

# Estilo CSS para melhorar a visualização e cores das métricas
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def carregar_e_treinar_modelo():
    try:
        # Coleta de dados resiliente
        df = yf.download("^BVSP", period="2y", interval="1d", progress=False)
        if df.empty: return None, None, None
        
        # Limpeza de MultiIndex
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
    
    # Target: Amanhã será maior que hoje?
    df['Fechamento_Amanha'] = df['Close'].shift(-1)
    df['Tendencia'] = (df['Fechamento_Amanha'] > df['Close']).astype(int)
    df = df.dropna()
    
    # Separação de Teste (30 dias)
    dias_teste = 30
    treino = df.iloc[:-dias_teste]
    teste = df.iloc[-dias_teste:]
    
    features = ['Retorno', 'Retorno_Ontem', 'Retorno_Anteontem', 'Dia_Semana', 'Variacao_Volume', 'Dist_Banda_Sup']
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(treino[features])
    X_teste = scaler.transform(teste[features])
    
    # Modelo
    modelo = GradientBoostingClassifier(learning_rate=0.2, max_depth=2, n_estimators=200, random_state=42)
    modelo.fit(X_treino, treino['Tendencia'])
    
    previsoes = modelo.predict(X_teste)
    acc = accuracy_score(teste['Tendencia'], previsoes)
    
    return teste, previsoes, acc

# Execução do Core
teste_df, previsoes_modelo, acuracia = carregar_e_treinar_modelo()

if teste_df is None:
    st.error("⚠️ Erro na conexão com o Yahoo Finance. Recarregue a página.")
    st.stop()

# ==========================================
# INTERFACE PRINCIPAL
# ==========================================

st.title("📟 Terminal de Inteligência Quantitativa | IBOVESPA")

# Barra Lateral com Filtros
st.sidebar.header("Painel de Controle")
filtro_tendencia = st.sidebar.multiselect(
    "Filtrar Tendência Real:",
    options=['Alta', 'Baixa'],
    default=['Alta', 'Baixa']
)

# --- BLOCO DE MÉTRICAS (Igual ao Investing/Warren) ---
col1, col2, col3, col4 = st.columns(4)

# Formatação brasileira: 178.488 em vez de 178,488.00
ultimo_valor = float(teste_df['Close'].iloc[-1])
col1.metric("Último IBOV", f"{ultimo_valor:,.0f}".replace(",", "."), "-0.49%")
col2.metric("Acurácia IA", f"{acuracia * 100:.1f}%", "Meta: 75%")
col3.metric("Janela de Teste", "30 Dias", "Último mês")
col4.metric("Status", "ONLINE", None)

st.divider()

# --- GRÁFICO INTERATIVO ---
st.subheader("Análise Gráfica de Sinais")
fig = go.Figure()
fig.add_trace(go.Scatter(x=teste_df.index, y=teste_df['Close'], name='Preço Fechamento', line=dict(color='#00CCFF', width=2)))

# Marcadores de Previsão
preds_color = ['#00FF00' if p == 1 else '#FF0000' for p in previsoes_modelo]
fig.add_trace(go.Scatter(
    x=teste_df.index, y=teste_df['Close'], mode='markers', name='Sinal IA',
    marker=dict(color=preds_color, size=10, symbol=['triangle-up' if p==1 else 'triangle-down' for p in previsoes_modelo])
))
fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# --- TABELA DE DADOS HISTÓRICOS (Estilo Investing) ---
st.subheader("Histórico de Previsões e Sinais")

# Preparação dos dados para a tabela
tabela_investing = pd.DataFrame({
    'Preço (Pontos)': teste_df['Close'].values.flatten(),
    'Real': ['Alta' if v == 1 else 'Baixa' for v in teste_df['Tendencia']],
    'Previsão IA': ['Alta' if v == 1 else 'Baixa' for v in previsoes_modelo]
}, index=teste_df.index)

# Aplicar filtros e inverter ordem (Mais recente no topo)
tabela_investing = tabela_investing[tabela_investing['Real'].isin(filtro_tendencia)]
tabela_investing = tabela_investing.sort_index(ascending=False)

# Função para colorir o texto (Verde para Alta, Vermelho para Baixa)
def colorir_texto(val):
    cor = '#00FF00' if 'Alta' in str(val) else '#FF0000'
    return f'color: {cor}; font-weight: bold'

# Formatação final e exibição
st.dataframe(
    tabela_investing.style
    .format({'Preço (Pontos)': "{:,.0f}"}, thousands=".", decimal=",")
    .applymap(colorir_texto, subset=['Real', 'Previsão IA']),
    use_container_width=True,
    height=500
)

# Rodapé profissional
st.markdown("---")
col_down1, col_down2 = st.columns([4,1])
with col_down1:
    st.caption("Terminal de dados em tempo real. Os sinais são baseados em algoritmos de Gradient Boosting.")
with col_down2:
    # Botão de download igual ao Investing
    csv = tabela_investing.to_csv().encode('utf-8')
    st.download_button("📥 Baixar Histórico", data=csv, file_name='historico_ibov.csv', mime='text/csv')