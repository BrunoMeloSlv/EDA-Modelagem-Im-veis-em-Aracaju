import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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


def extrair_bairro(address: str) -> str:
    if " - " in str(address):
        return str(address).split(" - ")[1].split(",")[0].strip()
    return str(address).split(",")[0].strip()


def limpar_bairro(bairro) -> str:
    if pd.isna(bairro):
        return bairro
    bairro = bairro.lower().strip()
    bairro = "".join(
        c for c in unicodedata.normalize("NFD", bairro)
        if unicodedata.category(c) != "Mn"
    )
    bairro = re.sub(r"zona de expans[ao]\s*", "", bairro)
    bairro = bairro.replace("-", " ").strip()
    correcoes = {"aruanda": "aruana"}
    bairro = correcoes.get(bairro, bairro)
    return bairro.title()


class AddressFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    A partir da coluna Address:
      1. Extrai e normaliza o Bairro
      2. Invalida bairros fora de Aracaju (fallback para renda mediana)
      3. Merge com a renda media do bairro
      4. Clip em Area
      5. Remove a coluna Address
    """
    def __init__(
        self,
        renda_map=None,
        renda_default=None,
        clip_area=460,
        bairros_invalidos=None,
    ):
        self.renda_map         = renda_map or {}
        self.renda_default     = renda_default or 0.0
        self.clip_area         = clip_area
        self.bairros_invalidos = bairros_invalidos or {"Se", "Ba"}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Bairro"] = X["Address"].apply(extrair_bairro).apply(limpar_bairro)
        X.loc[X["Bairro"].isin(self.bairros_invalidos), "Bairro"] = np.nan
        X["Renda (R$ aprox.)"] = (
            X["Bairro"].map(self.renda_map).fillna(self.renda_default)
        )
        X["Area"] = X["Area"].astype(float).clip(upper=self.clip_area)
        X = X.drop(columns=["Address"])
        return X
