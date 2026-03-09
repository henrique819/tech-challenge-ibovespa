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
# 1. CONFIGURAÇÃO DA PÁGINA & ESTILO
# ==========================================
st.set_page_config(page_title="Terminal IBOV - Tech Challenge", layout="wide", page_icon="📈")

# CSS para clonar o visual clean do Investing.com
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; color: #333; }
    [data-testid="stMetricValue"] { font-size: 32px; font-weight: 700; color: #1e222d; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 2px solid #e0e3eb; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-weight: 600; color: #70757a; }
    .stTabs [aria-selected="true"] { color: #2962FF !important; border-bottom: 2px solid #2962FF !important; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. COLETA E PROCESSAMENTO DE DADOS
# ==========================================
@st.cache_data(ttl=3600)
def get_full_data():
    try:
        # Coleta 2 anos para ter base de treino e cálculo de indicadores de longo prazo
        df = yf.download("^BVSP", period="2y", interval="1d", progress=False)
        if df.empty: return None
        
        # Correção de MultiIndex (Yahoo Finance)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        
        # Engenharia de Features
        df['Retorno'] = df['Close'].pct_change()
        df['Var_Vol'] = df['Volume'].pct_change()
        df['MM_20'] = df['Close'].rolling(window=20).mean()
        df['Std_20'] = df['Close'].rolling(window=20).std()
        df['Banda_Sup'] = df['MM_20'] + (df['Std_20'] * 2)
        df['Dist_Banda'] = (df['Close'] - df['Banda_Sup']) / df['Banda_Sup']
        df['Dia_Semana'] = df.index.dayofweek
        
        # Target: Amanhã > Hoje?
        df['Tendencia'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Limpeza blindada de NaNs e Infs (evita erro no Scaler)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df
    except:
        return None

# ==========================================
# 3. MODELO DE MACHINE LEARNING
# ==========================================
def treinar_modelo(df):
    if df is None: return None, None, 0
    
    features = ['Retorno', 'Dia_Semana', 'Var_Vol', 'Dist_Banda']
    dias_teste = 30 # Conforme enunciado do desafio
    
    treino = df.iloc[:-dias_teste].copy()
    teste = df.iloc[-dias_teste:].copy()
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(treino[features])
    X_ts = scaler.transform(teste[features])
    
    # Gradient Boosting Tunado
    modelo = GradientBoostingClassifier(learning_rate=0.2, max_depth=2, n_estimators=200, random_state=42)
    modelo.fit(X_tr, treino['Tendencia'])
    
    preds = modelo.predict(X_ts)
    acc = accuracy_score(teste['Tendencia'], preds)
    return teste, preds, acc

# Execução do Processamento
df_total = get_full_data()
teste_df, previsoes, acuracia = treinar_modelo(df_total)

if df_total is None:
    st.error("⚠️ Erro na conexão com dados financeiros. Recarregue a página.")
    st.stop()

# ==========================================
# 4. INTERFACE DO USUÁRIO (UI)
# ==========================================

# Header Principal
st.write(f"### Ibovespa (IBOV)")
col_h1, col_h2 = st.columns([1, 4])
with col_h1:
    v_atual = df_total['Close'].iloc[-1]
    v_ontem = df_total['Close'].iloc[-2]
    var_pct = ((v_atual / v_ontem) - 1) * 100
    # Formatação Brasileira 175.589
    st.metric(label="B3 - IBOV Realtime", value=f"{v_atual:,.0f}".replace(",", "."), delta=f"{var_pct:.2f}%")

# Sistema de Abas
aba_geral, aba_grafico, aba_historico = st.tabs(["Geral", "Gráfico Interativo", "Dados Históricos"])

# --- ABA GERAL ---
with aba_geral:
    st.markdown("#### Performance Acumulada")
    def ret_cor(d):
        v = ((df_total['Close'].iloc[-1] / df_total['Close'].iloc[-d]) - 1) * 100
        return f":{'green' if v > 0 else 'red'}[{v:.2f}%]"

    perf_data = pd.DataFrame({
        "Janela": ["1 Dia", "1 Semana", "1 Mês", "6 Meses", "1 Ano"],
        "Retorno": [ret_cor(2), ret_cor(6), ret_cor(22), ret_cor(126), ret_cor(252)]
    }).set_index("Janela")
    st.table(perf_data)
    
    st.markdown("#### Fundamentos de Hoje")
    c1, c2 = st.columns(2)
    c1.write(f"**Abertura:** {df_total['Open'].iloc[-1]:,.0f}".replace(",", "."))
    c1.write(f"**Fechamento Ant.:** {v_ontem:,.0f}".replace(",", "."))
    c2.write(f"**Máxima Diária:** {df_total['High'].iloc[-1]:,.0f}".replace(",", "."))
    c2.write(f"**Mínima Diária:** {df_total['Low'].iloc[-1]:,.0f}".replace(",", "."))

# --- ABA GRÁFICO (CANDLESTICK + VOLUME + ZOOM) ---
with aba_grafico:
    st.markdown("#### Terminal de Análise Técnica")
    
    # Seletor para despoluir o gráfico
    col_zoom, _ = st.columns([2, 3])
    with col_zoom:
        periodo = st.select_slider(
            "Visualizar período:",
            options=["1 Mês", "3 Meses", "6 Meses", "1 Ano", "Tudo"],
            value="3 Meses"
        )

    map_p = {"1 Mês": 22, "3 Meses": 66, "6 Meses": 126, "1 Ano": 252, "Tudo": len(df_total)}
    df_p = df_total.tail(map_p[periodo])
    teste_p = teste_df[teste_df.index >= df_p.index[0]]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=df_p.index, open=df_p['Open'], high=df_p['High'], 
        low=df_p['Low'], close=df_p['Close'], name='IBOV',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ), row=1, col=1)

    # 2. Sinais da IA (Triângulos)
    if not teste_p.empty:
        idx_start = len(teste_df) - len(teste_p)
        preds_p = previsoes[idx_start:]
        fig.add_trace(go.Scatter(
            x=teste_p.index, y=teste_p['High'] * 1.01, mode='markers', name='IA',
            marker=dict(color=['#00c853' if p == 1 else '#ff1744' for p in preds_p], 
                        size=11, symbol=['triangle-up' if p==1 else 'triangle-down' for p in preds_p],
                        line=dict(width=1, color='white'))
        ), row=1, col=1)

    # 3. Volume
    v_colors = ['#26a69a' if df_p['Close'].iloc[i] >= df_p['Open'].iloc[i] else '#ef5350' for i in range(len(df_p))]
    fig.add_trace(go.Bar(x=df_p.index, y=df_p['Volume'], marker_color=v_colors, opacity=0.8), row=2, col=1)

    # Ajustes de Eixos e Layout
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(tickformat=",.0f", row=1, col=1)
    fig.update_layout(template="plotly_white", height=600, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# --- ABA DADOS HISTÓRICOS ---
with aba_historico:
    st.markdown("#### Filtros de Movimento")
    f_mov = st.multiselect("Direção Real:", ["Alta", "Baixa"], default=["Alta", "Baixa"])
    
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

    # Botão de Download CSV
    csv = t_f.to_csv().encode('utf-8')
    st.download_button("📥 Baixar Histórico CSV", csv, "historico_ibovespa.csv", "text/csv")

st.markdown("---")
st.caption(f"Acurácia do Modelo: {acuracia*100:.2f}% | Algoritmo: Gradient Boosting Classifier")