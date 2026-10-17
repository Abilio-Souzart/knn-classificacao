"""Pipelines de pré-processamento e modelos.

Tudo que é aprendido dos dados (mediana, média/desvio, categorias de UF e,
quando usada, a seleção de variáveis) é ajustado apenas no treino ou nos
folds de treino da validação cruzada.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import COLS_CATEGORICAS, SEED


def signed_log(x):
    """sign(x) * log(1 + |x|): comprime caudas longas, aceita zero e negativos."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def make_preprocessor(X: pd.DataFrame, escala: str = "padronizada",
                      drop: str | None = None) -> ColumnTransformer:
    """escala: 'nenhuma' (só imputação), 'padronizada' (z-score) ou 'log' (log com sinal + z-score)."""
    numericas = [c for c in X.columns if c not in COLS_CATEGORICAS]
    passos = [SimpleImputer(strategy="median")]
    if escala == "log":
        passos.append(FunctionTransformer(signed_log, feature_names_out="one-to-one"))
    if escala in ("padronizada", "log"):
        passos.append(StandardScaler())
    # KNN usa as 3 colunas de UF (drop=None, como no notebook); árvore e logística
    # usam drop="first", como nos projetos anteriores, para reproduzir os mesmos números.
    cat = [OneHotEncoder(drop=drop, handle_unknown="ignore" if drop is None else "error",
                         sparse_output=False)]
    if escala != "nenhuma":
        cat.append(StandardScaler())
    return ColumnTransformer(
        [("num", make_pipeline(*passos), numericas),
         ("cat", make_pipeline(*cat), COLS_CATEGORICAS)],
        verbose_feature_names_out=False,
    )


def make_knn(X: pd.DataFrame, k: int, escala: str = "padronizada",
             n_variaveis: int | None = None) -> Pipeline:
    """KNN com pré-processamento. Se n_variaveis for informado, mantém só as
    n variáveis mais importantes de uma árvore de profundidade 4 ajustada no treino."""
    passos = [("prep", make_preprocessor(X, escala))]
    if n_variaveis:
        passos.append(("selecao", SelectFromModel(
            DecisionTreeClassifier(max_depth=4, random_state=SEED),
            max_features=n_variaveis, threshold=-np.inf)))
    passos.append(("modelo", KNeighborsClassifier(n_neighbors=k)))
    return Pipeline(passos)


def make_tree(X: pd.DataFrame, max_depth: int) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(X, "nenhuma", drop="first")),
                     ("modelo", DecisionTreeClassifier(max_depth=max_depth, random_state=SEED))])


def make_logit(X: pd.DataFrame, escala: str = "padronizada") -> Pipeline:
    return Pipeline([("prep", make_preprocessor(X, escala, drop="first")),
                     ("modelo", LogisticRegression(max_iter=5000, random_state=SEED))])


def variaveis_selecionadas(pipe: Pipeline) -> list[str]:
    nomes = pipe.named_steps["prep"].get_feature_names_out()
    return list(nomes[pipe.named_steps["selecao"].get_support()])
