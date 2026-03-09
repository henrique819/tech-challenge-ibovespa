# 📈 Previsão de Tendência IBOVESPA - Tech Challenge

Este projeto foi desenvolvido como parte do **Tech Challenge (Fase 2)** para um fundo de investimentos brasileiro. O objetivo é utilizar Machine Learning para prever a tendência do índice IBOVESPA (Alta ou Baixa) para o dia seguinte.

## 🚀 O Projeto
O modelo alcançou uma **acurácia de 76.67%** no conjunto de teste (últimos 30 dias), superando a meta estabelecida de 75%.

### 🔗 Acesse o Dashboard Interativo
> **[INSIRA AQUI O LINK DO SEU STREAMLIT DEPOIS DE PRONTO]**

---

## 🛠️ Storytelling Técnico

### 1. Aquisição e Exploração
Os dados foram coletados via API do Yahoo Finance (`yfinance`), cobrindo um período de **2 anos** de dados diários do IBOVESPA (`^BVSP`). Optamos por este intervalo para garantir que o modelo aprendesse com o regime econômico atual, evitando ruídos de crises muito antigas.

### 2. Engenharia de Atributos (Features)
Para aumentar o poder preditivo, foram criadas variáveis baseadas em análise técnica e comportamental:
* **Lags de Retorno:** Impacto do fechamento de D-1 e D-2.
* **Volatilidade e Volume:** Variação percentual do volume negociado.
* **Indicadores Técnicos:** Bandas de Bollinger (distância do preço para o topo) e RSI.
* **Sazonalidade:** Mapeamento do dia da semana.

### 3. O Modelo: Gradient Boosting Classifier
Após realizar um "torneio de algoritmos" entre Regressão Logística e Random Forest, o **Gradient Boosting** apresentou os melhores resultados. 
* **Justificativa:** Sua capacidade de corrigir erros em sequência (boosting) permitiu identificar tendências sutis que modelos lineares ignoraram.
* **Tuning:** O modelo foi ajustado com `max_depth=2` para evitar o *overfitting*, garantindo que ele aprenda a lógica do mercado e não apenas decore o histórico.

---

## ⚖️ Justificativa Técnica

* **Natureza Sequencial:** Tratada através de *features lagged* e janelas deslizantes (Moving Averages), transformando a série temporal em um problema de classificação supervisionada.
* **Trade-off Acurácia vs Overfitting:** Mantivemos árvores rasas para garantir a generalização do modelo. No mercado financeiro, um modelo muito complexo costuma falhar em dados novos; a simplicidade do nosso ajuste garantiu a estabilidade nos 30 dias de teste.

---

## 📁 Estrutura do Repositório
* `dashboard_ibov.py`: Script principal da aplicação web (Streamlit).
* `investing_completo.py`: Script de pesquisa, treinamento e validação do modelo.
* `requirements.txt`: Lista de dependências para o Deploy.
* `README.md`: Documentação completa do projeto.

## 👨‍💻 Desenvolvedor
Desenvolvido como parte do currículo de Data Science / BI & Analytics.