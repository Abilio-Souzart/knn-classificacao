"""Análises complementares à Questão 11: por que o KNN fica atrás e como melhorá-lo.

1. k além de 15, escolhido por validação cruzada.
2. Duas causas do desempenho fraco: caudas longas e variáveis irrelevantes.
3. Comparação final com árvore e logística, também por validação cruzada.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import K_AMPLO, N_FOLDS, SEED
from .preprocessing import make_knn, make_logit, make_tree, variaveis_selecionadas
from .utils import header, save_figure, save_table

CONFIGS = {
    "Padronizado, 22 variáveis (notebook)": dict(escala="padronizada"),
    "Log + padronizado, 22 variáveis": dict(escala="log"),
    "Log + padronizado, 2 variáveis": dict(escala="log", n_variaveis=2),
}


def folds():
    return StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)


def cv_scores(pipe, split) -> dict:
    r = cross_validate(pipe, split.X_tr, split.y_tr, cv=folds(), scoring=["accuracy", "roc_auc"])
    return {"acc_cv": r["test_accuracy"].mean(),
            "acc_cv_ep": r["test_accuracy"].std(ddof=1) / np.sqrt(N_FOLDS),
            "auc_cv": r["test_roc_auc"].mean()}


def curvas_cv(split) -> pd.DataFrame:
    linhas = []
    for nome, cfg in CONFIGS.items():
        for k in K_AMPLO:
            linhas.append({"config": nome, "k": k, **cv_scores(make_knn(split.X_tr, k, **cfg), split)})
    return pd.DataFrame(linhas)


def plot_curvas(curvas: pd.DataFrame, melhores: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    cores = ["#1565C0", "#FB8C00", "#2E7D32"]
    for cor, (nome, g) in zip(cores, curvas.groupby("config", sort=False)):
        ax.plot(g["k"], g["acc_cv"], "-", color=cor, lw=2, label=nome)
        ax.fill_between(g["k"], g["acc_cv"] - g["acc_cv_ep"], g["acc_cv"] + g["acc_cv_ep"],
                        color=cor, alpha=0.12)
        k = melhores.loc[nome, "k"]
        ax.plot(k, melhores.loc[nome, "acc_cv"], "o", color=cor, ms=9, mec="k")
    ax.axvspan(1, 15, color="grey", alpha=0.08, label="Faixa do enunciado (1 a 15)")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xlabel("k (número de vizinhos)")
    ax.set_ylabel("Acurácia (CV 10 folds, treino)")
    ax.set_title("KNN: acurácia de validação cruzada por k e pré-processamento")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "comp_cv_k_configuracoes")


def run(q11: dict) -> dict:
    header("ANÁLISES COMPLEMENTARES - Questão 11")
    split = q11["split"]

    curvas = curvas_cv(split)
    save_table(curvas.round(4), "comp_cv_por_k", index=False)
    melhores = curvas.loc[curvas.groupby("config", sort=False)["acc_cv"].idxmax()].set_index("config")

    linhas = []
    for nome, cfg in CONFIGS.items():
        k = int(melhores.loc[nome, "k"])
        pipe = make_knn(split.X_tr, k, **cfg).fit(split.X_tr, split.y_tr)
        prob = pipe.predict_proba(split.X_te)[:, 1]
        extra = ", ".join(variaveis_selecionadas(pipe)) if cfg.get("n_variaveis") else "todas"
        linhas.append({"config": nome, "k (CV)": k, "acc CV": melhores.loc[nome, "acc_cv"],
                       "AUC CV": melhores.loc[nome, "auc_cv"],
                       "acc teste": accuracy_score(split.y_te, prob >= 0.5),
                       "AUC teste": roc_auc_score(split.y_te, prob), "variáveis": extra})
    resumo = pd.DataFrame(linhas).set_index("config")
    save_table(resumo.round(4), "comp_knn_configuracoes")
    print("1-2. Melhor k por validação cruzada em cada configuração:")
    print(resumo.round(3).to_string())
    plot_curvas(curvas, melhores)

    # 3. Comparação final por CV
    knn_best = resumo["acc CV"].idxmax()
    cfg, k = CONFIGS[knn_best], int(resumo.loc[knn_best, "k (CV)"])
    modelos = {
        f"KNN ({knn_best.lower()}, k = {k})": make_knn(split.X_tr, k, **cfg),
        "KNN padronizado, k = 15 (notebook)": make_knn(split.X_tr, 15),
        "Árvore profundidade 2": make_tree(split.X_tr, 2),
        "Árvore profundidade 4": make_tree(split.X_tr, 4),
        "Logística padronizada": make_logit(split.X_tr, "padronizada"),
        "Logística com log": make_logit(split.X_tr, "log"),
    }
    linhas = []
    for nome, m in modelos.items():
        s = cv_scores(m, split)
        m.fit(split.X_tr, split.y_tr)
        prob = m.predict_proba(split.X_te)[:, 1]
        linhas.append({"modelo": nome, "acc CV": s["acc_cv"], "AUC CV": s["auc_cv"],
                       "acc teste": accuracy_score(split.y_te, prob >= 0.5),
                       "AUC teste": roc_auc_score(split.y_te, prob)})
    final = pd.DataFrame(linhas).set_index("modelo")
    save_table(final.round(4), "comp_modelos_final")
    print("\n3. Comparação final (CV 10 folds no treino e teste):")
    print(final.round(3).to_string())
    return {"resumo": resumo, "final": final}
