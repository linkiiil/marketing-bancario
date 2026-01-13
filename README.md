# Marketing Bancário
🏦 IA de Propensão Bancária: Triagem de Leads e Depósitos a Prazo

Este repositório contém o ciclo completo de ciência de dados — desde a análise exploratória (EDA) até a implantação de uma aplicação interativa — para prever a propensão de clientes contratarem depósitos a prazo em uma instituição bancária portuguesa.

📖 Contexto e Objetivos
Campanhas de marketing direto enfrentam baixas taxas de conversão (apenas 11,7% neste dataset). O objetivo deste projeto é otimizar o retorno sobre o investimento (ROI) através de:

Triagem Inteligente (Ranking): Identificar o "Top 10%" de clientes com maior chance de conversão, onde o modelo atinge um Lift de até 5.89x.

Interpretabilidade com SHAP: Explicar quais fatores (como saldo ou idade) influenciam cada score individual.

Prevenção de Fadiga: Sinalizar clientes com excesso de contatos (campaign > 20), onde a probabilidade de conversão cai drasticamente.

🛠️ Estrutura do Repositório

Projeto - Dados Bancários.pdf: Documentação completa contendo hipóteses, análise exploratória, tratamento de data leakage e métricas de desempenho dos modelos.

bank_marketing.py: Aplicação Streamlit que serve como interface para o usuário final (gerentes de conta), permitindo simulações em tempo real.

modelo_bank_marketing_xgboost.pkl: Pipeline final serializado, contendo o pré-processador (ColumnTransformer) e o modelo XGBoost tunado.

requirements.txt: Lista de dependências necessárias para executar o projeto (Streamlit, Scikit-learn, XGBoost, SHAP etc.).

🔬 Principais Achados e Hipóteses

Liquidez (H2): Clientes com saldos médios anuais (balance) positivos têm maior propensão ao investimento.

Barreiras de Crédito (H3): A existência de empréstimo habitacional (housing) atua como um redutor consistente na probabilidade de conversão.

Perfil Demográfico (H4): Indivíduos entre 60 e 80 anos demonstram maior interesse em produtos de baixo risco.

Efeito Sazonal: A análise via Heatmap revelou que os contatos realizados no Q1 (primeiro trimestre) apresentam taxas de conversão superiores em janelas específicas de dias.

🚀 Como Executar

1. Pré-requisitos
Certifique-se de ter o Python 3.9+ instalado. Instale as dependências:

Bash

pip install -r requirements.txt

2. Executando o Dashboard (Streamlit)
Para iniciar a aplicação interativa e realizar predições:

Bash

streamlit run bank_marketing.py

📊 Avaliação do Modelo
O modelo escolhido foi o XGBoost, otimizado via HalvingRandomSearchCV focado em Average Precision (AP).

Lift @ 10%: Captura ~38% de todas as conversões reais abordando apenas os 10% melhores leads.

ROC-AUC: 0.7628 (Embora o foco tenha sido métricas de ranking como Precision@K e Lift@K).

Nota técnica: Foram removidas as variáveis duration, contact e poutcome do treinamento para evitar Data Leakage, garantindo que o modelo seja utilizável em cenários onde o desfecho da ligação ainda não ocorreu.
