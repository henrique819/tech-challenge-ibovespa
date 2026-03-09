import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Terminal IBOV - Elite", layout="wide", page_icon="📈")

# Estilo CSS Investing Style
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; color: #333; }
    [data-testid="stMetricValue"] { font-size: 32px; font-weight: 700; color: #1e222d; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 2px solid #e0e3eb; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-weight: 600; color: #70757a; }
    .stTabs [aria-selected="true"] { color: #2962FF !important; border-bottom: 2px solid #2962FF !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_full_data():
    try:
        # Baixa os dados
        df = yf.download("^BVSP", period="2y", interval="1d", progress=False)
        if df.empty: return None
        
        # Limpa nomes de colunas (MultiIndex fix)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Engenharia de Features
        df['Retorno'] = df['Close'].pct_change()
        df['Var_Vol'] = df['Volume'].pct_change()
        df['MM_20'] = df['Close'].rolling(window=20).mean()
        df['Std_20'] = df['Close'].rolling(window=20).std()
        df['Banda_Sup'] = df['MM_20'] + (df['Std_20'] * 2)
        df['Dist_Banda'] = (df['Close'] - df['Banda_Sup']) / df['Banda_Sup']
        df['Dia'] = df.index.dayofweek
        
        # Target (Amanhã será maior que Hoje?)
        df['Tendencia'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # --- LIMPEZA BLINDADA ---
        # Substitui infinitos por NaN e depois dropa tudo que for NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        return df
    except:
        return None

df_total = get_full_data()

def treinar_modelo(df):
    if df is None: return None, None, 0
    
    features = ['Retorno', 'Dia', 'Var_Vol', 'Dist_Banda']
    dias_teste = 30
    
    # Separação
    treino = df.iloc[:-dias_teste].copy()
    teste = df.iloc[-dias_teste:].copy()
    
    # Normalização
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(treino[features])
    X_ts = scaler.transform(teste[features])
    
    # Modelo
    modelo = GradientBoostingClassifier(learning_rate=0.2, max_depth=2, n_estimators=200, random_state=42)
    modelo.fit(X_tr, treino['Tendencia'])
    
    preds = modelo.predict(X_ts)
    acc = accuracy_score(teste['Tendencia'], preds)
    return teste, preds, acc

# Execução
teste_df, previsoes, acc = treinar_modelo(df_total)

if df_total is None:
    st.error("⚠️ Erro crítico ao processar dados. Tente recarregar.")
    st.stop()

# ==========================================
# HEADER
# ==========================================
st.write(f"### Ibovespa (IBOV)")
col_h1, col_h2 = st.columns([1, 4])
with col_h1:
    v_atual = df_total['Close'].iloc[-1]
    v_ontem = df_total['Close'].iloc[-2]
    var_pct = ((v_atual / v_ontem) - 1) * 100
    st.metric(label="B3 - IBOV Realtime", value=f"{v_atual:,.0f}".replace(",", "."), delta=f"{var_pct:.2f}%")

aba_geral, aba_grafico, aba_historico = st.tabs(["Geral", "Gráfico Interativo", "Dados Históricos"])

# --- ABA GERAL ---
with aba_geral:
    st.markdown("#### Performance Acumulada")
    def ret_cor(d):
        v = ((df_total['Close'].iloc[-1] / df_total['Close'].iloc[-d]) - 1) * 100
        return f":{'green' if v > 0 else 'red'}[{v:.2f}%]"

    st.table(pd.DataFrame({
        "Janela": ["1D", "1 Sem", "1 Mês", "6 Meses", "1 Ano"],
        "Retorno": [ret_cor(2), ret_cor(6), ret_cor(22), ret_cor(126), ret_cor(252)]
    }).set_index("Janela"))
    
    c1, c2 = st.columns(2)
    c1.write(f"**Abertura:** {df_total['Open'].iloc[-1]:,.0f}".replace(",", "."))
    c1.write(f"**Fechamento Ant.:** {v_ontem:,.0f}".replace(",", "."))
    c2.write(f"**Máxima:** {df_total['High'].iloc[-1]:,.0f}".replace(",", "."))
    c2.write(f"**Mínima:** {df_total['Low'].iloc[-1]:,.0f}".replace(",", "."))

# --- ABA GRÁFICO (CANDLESTICKS + VOLUME) ---
with aba_grafico:
    st.markdown("#### Terminal Avançado de Velas e Volume")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_total.index, open=df_total['Open'], high=df_total['High'], 
        low=df_total['Low'], close=df_total['Close'], name='IBOV',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ), row=1, col=1)

    # Marcadores da IA
    sinais_y = teste_df['High'] * 1.02
    fig.add_trace(go.Scatter(
        x=teste_df.index, y=sinais_y, mode='markers', name='Previsão',
        marker=dict(color=['#008000' if p == 1 else '#FF0000' for p in previsoes], 
                    size=10, symbol=['triangle-up' if p==1 else 'triangle-down' for p in previsoes])
    ), row=1, col=1)

    # Volume
    vol_colors = ['#26a69a' if df_total['Close'].iloc[i] >= df_total['Open'].iloc[i] else '#ef5350' 
                  for i in range(len(df_total))]
    fig.add_trace(go.Bar(x=df_total.index, y=df_total['Volume'], name='Volume', marker_color=vol_colors), row=2, col=1)

    fig.update_layout(template="white", xaxis_rangeslider_visible=False, height=650, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# --- ABA DADOS HISTÓRICOS ---
with aba_historico:
    st.markdown("#### Histórico Formatado (Investing Style)")
    f_mov = st.multiselect("Direção:", ["Alta", "Baixa"], default=["Alta", "Baixa"])
    
    t_f = pd.DataFrame({
        'Preço (Pontos)': teste_df['Close'],
        'Real': ['Alta' if v == 1 else 'Baixa' for v in teste_df['Tendencia']],
        'Previsão IA': ['Alta' if v == 1 else 'Baixa' for v in previsoes]
    }).sort_index(ascending=False)
    
    t_f = t_f[t_f['Real'].isin(f_mov)]
    t_f.index = t_f.index.strftime('%d/%m/%Y')

    st.dataframe(
        t_f.style
        .format({'Preço (Pontos)': "{:,.0f}"}, thousands=".", decimal=",")
        .applymap(lambda x: f"color: {'#008000' if x=='Alta' else '#d91e18'}; font-weight: bold", 
                  subset=['Real', 'Previsão IA']),
        use_container_width=True, height=500
    )