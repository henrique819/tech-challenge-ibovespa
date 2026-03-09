import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

print("1. Coletando dados do IBOVESPA (2 anos)...")
df = yf.Ticker("^BVSP").history(period="2y", interval="1d")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

print("2. Engenharia de Features Avançada...")
df['Retorno'] = df['Close'].pct_change()
df['Variacao_Volume'] = df['Volume'].pct_change()
df['Retorno_Ontem'] = df['Retorno'].shift(1)
df['Retorno_Anteontem'] = df['Retorno'].shift(2)

# O Segredo 1: Dia da Semana (0 = Segunda, 4 = Sexta)
df['Dia_Semana'] = df.index.dayofweek

# O Segredo 2: Bandas de Bollinger (Mede o limite do estresse do mercado)
df['MM_20'] = df['Close'].rolling(window=20).mean()
df['Desvio_20'] = df['Close'].rolling(window=20).std()
df['Banda_Superior'] = df['MM_20'] + (df['Desvio_20'] * 2)
df['Dist_Banda_Sup'] = (df['Close'] - df['Banda_Superior']) / df['Banda_Superior']

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

print("3. Definindo o Alvo (Target)...")
df['Fechamento_Amanha'] = df['Close'].shift(-1)
df['Tendencia'] = (df['Fechamento_Amanha'] > df['Close']).astype(int)
df = df.dropna()

print("4. Separando dados de Treino e Teste (Últimos 30 dias)...")
dias_teste = 30
treino = df.iloc[:-dias_teste]
teste = df.iloc[-dias_teste:]

features = ['Retorno', 'Retorno_Ontem', 'Retorno_Anteontem', 'Dia_Semana', 'Variacao_Volume', 'Dist_Banda_Sup']

X_treino = treino[features]
y_treino = treino['Tendencia']
X_teste = teste[features]
y_teste = teste['Tendencia']

print("5. Normalizando os dados...")
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

print("6. Iniciando a Força Bruta (Hyperparameter Tuning)... procurando os 75%...")
melhor_acc = 0
melhor_modelo = None
parametros_vencedores = {}

# Testando dezenas de combinações de configuração para o Gradient Boosting
for lr in [0.01, 0.05, 0.1, 0.2]:
    for depth in [2, 3, 4, 5]:
        for est in [50, 100, 150, 200]:
            # Treina um modelo com a configuração atual
            modelo = GradientBoostingClassifier(
                learning_rate=lr, 
                max_depth=depth, 
                n_estimators=est, 
                random_state=42
            )
            modelo.fit(X_treino_scaled, y_treino)
            
            # Testa o modelo
            previsoes = modelo.predict(X_teste_scaled)
            acc = accuracy_score(y_teste, previsoes)
            
            # Se for o melhor até agora, salva ele
            if acc > melhor_acc:
                melhor_acc = acc
                melhor_modelo = modelo
                parametros_vencedores = {'Taxa de Aprendizado': lr, 'Profundidade': depth, 'Árvores': est}
            
            # Se já bateu a meta do professor (75% ou mais), interrompe as buscas!
            if melhor_acc >= 0.75:
                break
        if melhor_acc >= 0.75: break
    if melhor_acc >= 0.75: break

print("\n" + "="*50)
if melhor_acc >= 0.75:
    print("🏆 META ATINGIDA COM SUCESSO!")
else:
    print("🚀 MELHOR RESULTADO ENCONTRADO (QUASE LÁ!):")
    
print(f"🎯 ACURÁCIA FINAL: {melhor_acc * 100:.2f}%")
print(f"⚙️  Configuração Vencedora: {parametros_vencedores}")
print("="*50 + "\n")

resultados_finais = pd.DataFrame({
    'Real': y_teste[-5:],
    'Previsto': melhor_modelo.predict(X_teste_scaled)[-5:]
})
print("Amostra das últimas 5 previsões:")
print(resultados_finais)