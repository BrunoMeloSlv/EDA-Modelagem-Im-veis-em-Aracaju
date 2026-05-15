import streamlit as st
from utils import AddressFeatureBuilder, limpar_bairro, extrair_bairro, limpar_numerico, limpar_preco
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
import unicodedata

st.set_page_config(
    page_title="Imóveis Aracaju",
    page_icon="🏠",
    layout="wide",
)

# ── Carrega artefatos ─────────────────────────────────────────────────────────

@st.cache_resource
def carregar_modelos():
    pipeline = joblib.load("pipeline.pkl")
    model    = joblib.load("model.pkl")
    meta     = joblib.load("meta.pkl")
    return pipeline, model, meta

pipeline, model, meta = carregar_modelos()


# ── Carrega dataset para EDA ──────────────────────────────────────────────────

@st.cache_data
def carregar_dados():
    import kagglehub
    path = kagglehub.dataset_download("luccabortoloso2/house-princing-aracaju-brazil")
    df   = pd.read_csv(f"{path}/house_prices_aracaju_v2.csv")
    df   = df.replace("\n", "", regex=True)

    def limpar_preco(serie):
        return (
            serie
            .str.replace(r"R\$ ", "", regex=True)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.extract(r"(\d+(?:\.\d+)?)", expand=False)
            .astype(float)
        )

    def limpar_numerico(serie):
        return pd.to_numeric(
            serie.astype(str).str.strip().str.replace(r"[^\d.]", "", regex=True),
            errors="coerce"
        )

    def extrair_bairro(address):
        if " - " in str(address):
            return str(address).split(" - ")[1].split(",")[0].strip()
        return str(address).split(",")[0].strip()

    def limpar_bairro(bairro):
        if pd.isna(bairro):
            return bairro
        bairro = bairro.lower().strip()
        bairro = "".join(
            c for c in unicodedata.normalize("NFD", bairro)
            if unicodedata.category(c) != "Mn"
        )
        bairro = re.sub(r"zona de expans[ao]\s*", "", bairro)
        return bairro.replace("-", " ").strip().title()

    df["Price"] = limpar_preco(df["Price"])
    for col in ["Area", "Rooms", "Bathrooms", "Garage Cars"]:
        df[col] = limpar_numerico(df[col])
    df["Garage Cars"] = df["Garage Cars"].fillna(0)
    df["Bairro"] = df["Address"].apply(extrair_bairro).apply(limpar_bairro)
    df["Renda (R$ aprox.)"] = df["Bairro"].map(meta["renda_bairro"]).fillna(meta["renda_mediana"])
    df = df.dropna(subset=["Price", "Area", "Rooms", "Bathrooms"])
    df = df[~df["Bairro"].isin({"Se", "Ba"})]
    return df

df = carregar_dados()


# ── Estilo ────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid")
plt.rcParams["axes.spines.top"]   = False
plt.rcParams["axes.spines.right"] = False
COR = "#378ADD"


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🏠 Imóveis em Aracaju")
st.caption("Análise exploratória e previsão de preço · Random Forest + log(y) · R² 0.82")
st.divider()


# ── Abas ──────────────────────────────────────────────────────────────────────

aba_eda, aba_previsao = st.tabs(["📊 Análise Exploratória", "🔮 Previsão de Preço"])


# ════════════════════════════════════════════════════════════════════════════
# ABA 1 — EDA
# ════════════════════════════════════════════════════════════════════════════

with aba_eda:

    # Métricas rápidas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Imóveis no dataset", f"{len(df):,}")
    col2.metric("Preço mediano", f"R$ {df['Price'].median():,.0f}")
    col3.metric("Área mediana", f"{df['Area'].median():.0f} m²")
    col4.metric("Bairros mapeados", df["Bairro"].nunique())

    st.divider()

    # Distribuição de preços
    st.subheader("Distribuição dos Preços")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df["Price"], bins=40, kde=True, color=COR, ax=axes[0])
    axes[0].set_xlabel("Preço (R$)")
    axes[0].set_ylabel("Frequência")
    axes[0].set_title("Histograma")

    sns.boxplot(x=df["Price"], color=COR, ax=axes[1])
    axes[1].set_xlabel("Preço (R$)")
    axes[1].set_title("Boxplot — outliers à direita")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption(
        f"Média: R$ {df['Price'].mean():,.0f} | "
        f"Mediana: R$ {df['Price'].median():,.0f} | "
        f"Assimetria: {df['Price'].skew():.2f} — distribuição assimétrica à direita"
    )

    st.divider()

    # Preço por bairro
    st.subheader("Preço Mediano por Bairro")

    min_imoveis = st.slider(
        "Mínimo de imóveis por bairro", 5, 50, 15,
        help="Filtra bairros com poucos imóveis para evitar distorções"
    )

    preco_bairro = (
        df.groupby("Bairro")
        .filter(lambda x: len(x) >= min_imoveis)
        .groupby("Bairro")["Price"]
        .median()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, max(4, len(preco_bairro) * 0.35)))
    preco_bairro.plot(kind="barh", color=COR, ax=ax)
    ax.set_xlabel("Preço Mediano (R$)")
    ax.set_title(f"Bairros com ao menos {min_imoveis} imóveis")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Scatter preço vs renda
    st.subheader("Preço × Renda Média do Bairro")

    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(
        df["Renda (R$ aprox.)"],
        df["Price"],
        alpha=0.35, color=COR, s=25, edgecolors="none"
    )
    ax.set_xlabel("Renda Média do Bairro (R$)")
    ax.set_ylabel("Preço do Imóvel (R$)")
    ax.set_title("Correlação positiva — bairros mais ricos, imóveis mais caros")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Distribuição das features
    st.subheader("Distribuição das Features")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for ax, col in zip(axes, ["Area", "Rooms", "Bathrooms", "Garage Cars"]):
        sns.histplot(df[col], bins=25, kde=True, color=COR, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Heatmap correlação
    st.subheader("Correlação entre Variáveis")

    num_cols_corr = ["Area", "Rooms", "Bathrooms", "Garage Cars", "Renda (R$ aprox.)", "Price"]
    corr = df[num_cols_corr].corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
        square=True, ax=ax
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# ABA 2 — PREVISÃO
# ════════════════════════════════════════════════════════════════════════════

with aba_previsao:

    st.subheader("Preencha as características do imóvel")

    col_form, col_resultado = st.columns([1, 1], gap="large")

    with col_form:
        bairro = st.selectbox(
            "Bairro",
            options=sorted(meta["bairros"]),
            help="Selecione o bairro do imóvel"
        )
        area = st.number_input(
            "Área (m²)", min_value=20, max_value=1000, value=120, step=10
        )
        quartos = st.number_input(
            "Quartos", min_value=1, max_value=10, value=3, step=1
        )
        banheiros = st.number_input(
            "Banheiros", min_value=1, max_value=10, value=2, step=1
        )
        vagas = st.number_input(
            "Vagas de garagem", min_value=0, max_value=10, value=1, step=1
        )

        prever = st.button("Prever preço", type="primary", use_container_width=True)

    with col_resultado:
        if prever:
            # Monta entrada no mesmo formato que o pipeline espera
            renda_bairro = meta["renda_bairro"].get(bairro, meta["renda_mediana"])
            area_clipped = min(float(area), meta["clip_area"])

            entrada = pd.DataFrame([{
                "Area":               area_clipped,
                "Rooms":              quartos,
                "Bathrooms":          banheiros,
                "Garage Cars":        vagas,
                "Renda (R$ aprox.)":  renda_bairro,
                # Address ficticio — o feature_builder vai extrair o bairro dele
                # mas como ja temos as features prontas, bypassamos com um truque:
                # passamos o bairro diretamente na coluna Bairro apos o builder
            }])

            # Como o pipeline espera "Address", criamos um endereço fake
            # que o extrair_bairro vai ignorar — a renda ja esta no DataFrame
            # A solução mais limpa: criar uma versao do builder que aceita
            # o DataFrame ja processado. Fazemos isso aqui diretamente:
            from sklearn.preprocessing import StandardScaler
            scaler = pipeline.named_steps["preprocessor"].named_transformers_["num"]
            X_entrada = scaler.transform(entrada[meta["feat_num"]])

            pred_log  = model.predict(X_entrada)[0]
            preco_est = np.expm1(pred_log)

            st.metric(
                label="Preço estimado",
                value=f"R$ {preco_est:,.0f}",
            )

            # Posição no mercado
            y_todos = meta["y_train"]
            percentil = (y_todos < preco_est).mean() * 100

            st.caption(
                f"Este imóvel está acima de **{percentil:.0f}%** dos imóveis "
                f"do dataset — bairro {bairro}, renda média R$ {renda_bairro:,.0f}/mês"
            )

            # Histograma com posição do imóvel
            fig, ax = plt.subplots(figsize=(7, 3))
            sns.histplot(y_todos, bins=40, color="#B4B2A9", kde=True, ax=ax)
            ax.axvline(preco_est, color="#E24B4A", linewidth=2, linestyle="--",
                       label=f"Sua estimativa: R$ {preco_est:,.0f}")
            ax.set_xlabel("Preço (R$)")
            ax.set_ylabel("Frequência")
            ax.set_title("Posição no mercado")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Feature importance
            st.subheader("O que mais pesou nesta previsão")
            importances = pd.Series(
                model.feature_importances_,
                index=meta["feat_num"]
            ).sort_values(ascending=True)

            fig, ax = plt.subplots(figsize=(7, 3))
            importances.plot(kind="barh", color=COR, ax=ax)
            ax.set_xlabel("Importância")
            ax.set_title("Feature importance — modelo global")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        else:
            st.info("Preencha os dados à esquerda e clique em **Prever preço**.")
