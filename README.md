# 🏠 Previsão de Preços de Imóveis em Aracaju

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Production-success)

Projeto completo de **Análise de Dados + Machine Learning** para previsão de preços de imóveis em Aracaju (SE), com aplicação interativa em **Streamlit**.

---

## 📌 Objetivo

Construir um modelo capaz de prever o preço de imóveis com base em características como:

- Área  
- Quartos  
- Banheiros  
- Vagas de garagem  
- Bairro (proxy via renda média)  

---

## 📊 Dataset

Fonte: Kaggle  
Dataset: *House Pricing Aracaju - Brazil*

Principais variáveis:

- `Price`  
- `Area`  
- `Rooms`  
- `Bathrooms`  
- `Garage Cars`  
- `Address`  

---

## 🧹 Tratamento de Dados

Foi realizado:

- Limpeza de preços (`R$`, separadores, etc)  
- Conversão de variáveis numéricas  
- Extração de bairro a partir do endereço  
- Padronização de texto (acentos, caixa, etc)  
- Criação de feature: **Renda média do bairro**  
- Remoção de valores nulos  
- Remoção de bairros inconsistentes  

---

## 🧠 Feature Engineering

Principais features utilizadas:

- Área (com clipping)  
- Quartos  
- Banheiros  
- Vagas  
- Renda média do bairro  

---

## 🤖 Modelagem

Pipeline:

1. Pré-processamento (StandardScaler)  
2. Modelo: **Random Forest Regressor**  
3. Transformação do target:

```python
log(Price)
```

--

## 📈 Resultados
R²: 0.82
Boa capacidade de generalização
Redução do impacto de outliers com log(y)

--

## 🔍 Análise Exploratória (EDA)

Inclui:

Distribuição de preços
Boxplot (outliers)
Preço por bairro
Correlação entre variáveis
Heatmap

Insights:
Forte correlação entre renda do bairro e preço
Área é uma das variáveis mais relevantes

--

## 🚀 Aplicação (Streamlit)

O projeto conta com um app interativo com:

📊 Aba 1 — EDA
Métricas gerais
Histogramas
Boxplots
Gráficos por bairro
Heatmap
🔮 Aba 2 — Previsão

Usuário informa:

Bairro
Área
Quartos
Banheiros
Vagas

Retorno:

💰 Preço estimado
📊 Percentil no mercado
📈 Distribuição comparativa
🔍 Importância das variáveis

## ⚙️ Estrutura do Projeto

├── app.py
├── pipeline.pkl
├── model.pkl
├── meta.pkl
├── notebook.ipynb
└── utils.py


Distribuição assimétrica
