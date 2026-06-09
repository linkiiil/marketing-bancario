# Marketing Bancário

### 🏦 Previsão de Adesão a Depósitos a Prazo

Este projeto aplica técnicas de Machine Learning para prever a probabilidade de um cliente aderir a um depósito a prazo após uma campanha de marketing bancário. Além do desenvolvimento do modelo preditivo, foi criada uma aplicação interativa em Streamlit para apoiar a priorização de clientes e a interpretação das previsões.

### 📖 Contexto

Campanhas de marketing bancário costumam apresentar baixas taxas de conversão, tornando importante identificar quais clientes possuem maior potencial de contratação. Utilizando dados demográficos, financeiros e históricos de campanhas, este projeto busca transformar informações dos clientes em suporte à tomada de decisão.

### 🎯 Objetivos
Prever a probabilidade de adesão a depósitos a prazo.
Comparar diferentes algoritmos de classificação.
Identificar os clientes com maior potencial de conversão.
Interpretar as previsões utilizando SHAP.
Disponibilizar uma aplicação interativa para simulações e apoio à decisão.

### 📂 Estrutura do Repositório

├── Projeto - Dados Bancários.ipynb: desenvolvimento completo do projeto, incluindo análise exploratória, pré-processamento, modelagem e interpretabilidade.

├── bank_marketing.py: aplicação Streamlit para realização de previsões e análise dos resultados.

├── modelo_bank_marketing_xgboost.pkl: pipeline final contendo pré-processamento e modelo treinado.

├── gráficos/: gráficos de avaliação e desempenho dos modelos.

├── requirements.txt: dependências necessárias para execução do projeto.

### 📊 Avaliação do Modelo

O desempenho dos modelos foi avaliado sob diferentes perspectivas, incluindo:

Curva ROC-AUC
Lift Chart
Matriz de Confusão
Brier Score
Separabilidade de Classes

### 💡 Principais Insights

Clientes com saldos mais elevados tendem a apresentar maior propensão à conversão.
A presença de empréstimos habitacionais está associada a menores taxas de adesão.
O histórico de campanhas anteriores possui influência relevante no comportamento dos clientes.
Existem padrões sazonais que impactam o desempenho das campanhas.

### 🚀 Aplicação Streamlit

A aplicação permite:

Simular perfis de clientes.
Calcular scores de propensão à conversão.
Priorizar leads com maior potencial.
Visualizar explicações individuais por meio de SHAP.
Apoiar decisões de marketing de forma intuitiva.

### ⚙️ Como Executar

Instale as dependências:

pip install -r requirements.txt

Execute a aplicação:

streamlit run bank_marketing.py

### 👤 Autor

Rodrigo Emanuel Freitas Losada
