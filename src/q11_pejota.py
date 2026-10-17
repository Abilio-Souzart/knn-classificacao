"""Questão 11: KNN para inadimplência de empresas (base PEJOTA)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             roc_auc_score)

from .config import K_ENUNCIADO
from .data import load_pejota, make_split
from .preprocessing import make_knn, make_logit, make_preprocessor, make_tree
from .utils import header, save_figure, save_table


def participacao_na_distancia(X_tr: pd.DataFrame, escala: str) -> pd.Series:
    """Quanto cada variável contribui, em média, para a distância euclidiana ao quadrado.

    E[(x_i - x_j)^2] = 2 Var(x) para pares aleatórios, então a participação de
    cada variável é proporcional à sua variância depois do pré-processamento.
    """
    prep = make_preprocessor(X_tr, escala).fit(X_tr)
    Z = pd.DataFrame(prep.transform(X_tr), columns=prep.get_feature_names_out())
    v = Z.var()
    return (v / v.sum()).sort_values(ascending=False)


def plot_padronizacao(acc_raw, acc_sc, part_raw) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1, 1.6]})
    bars = axes[0].bar(["Sem padronização", "Com padronização"], [acc_raw, acc_sc],
                       color=["#F44336", "#2196F3"], width=0.5)
    axes[0].bar_label(bars, labels=[f"{v:.1%}" for v in (acc_raw, acc_sc)], padding=3)
    axes[0].axhline(0.5, color="grey", ls="--", lw=1, label="Baseline (50%)")
    axes[0].set_ylim(0.4, 0.75)
    axes[0].set_ylabel("Acurácia no teste")
    axes[0].set_title("(b) KNN k = 5")
    axes[0].legend()
    top = part_raw.head(6)[::-1]
    axes[1].barh(top.index, 100 * top.values, color="#F44336")
    for i, v in enumerate(top.values):
        axes[1].text(100 * v + 0.5, i, f"{100 * v:.1f}%", va="center", fontsize=9)
    axes[1].set_xlabel("% da distância euclidiana (sem padronização)")
    axes[1].set_title("Sem padronização, duas variáveis definem os vizinhos")
    axes[1].set_xlim(0, 105)
    fig.tight_layout()
    save_figure(fig, "q11b_padronizacao")


def plot_curva_k(curva: pd.DataFrame, melhor: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curva.index, curva["acc_treino"], "o-", color="#90CAF9", label="Treino")
    ax.plot(curva.index, curva["acc_teste"], "s-", color="#1565C0", lw=2, label="Teste")
    ax.axvline(melhor, color="grey", ls="--", label=f"Maior acurácia no teste: k = {melhor}")
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="Baseline (50%)")
    ax.set_xticks(curva.index)
    ax.set_xlabel("k (número de vizinhos)")
    ax.set_ylabel("Acurácia")
    ax.set_title("(c) KNN padronizado: acurácia por k")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "q11c_curva_k")


def plot_matriz(cm, k, acc) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay(cm, display_labels=["BOM (0)", "MAU (1)"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(f"(d) Matriz de confusão, k = {k}\nacurácia no teste = {acc:.1%}")
    fig.tight_layout()
    save_figure(fig, "q11d_matriz_confusao")


def avaliar(pipe, split) -> dict:
    prob = pipe.predict_proba(split.X_te)[:, 1]
    return {"acc_teste": accuracy_score(split.y_te, prob >= 0.5),
            "AUC_teste": roc_auc_score(split.y_te, prob)}


def run() -> dict:
    header("QUESTÃO 11 - KNN na base PEJOTA")
    df = load_pejota()
    split = make_split(df)

    # (a)
    ausentes = df.isna().sum()
    print(f"(a) {len(df)} empresas, {df['DEFAULT'].mean():.0%} inadimplentes")
    print("    Ausentes antes da imputação: " +
          ", ".join(f"{c} ({n})" for c, n in ausentes[ausentes > 0].items()))
    print(f"    Treino: {len(split.y_tr)} | Teste: {len(split.y_te)} | "
          f"% MAU: {split.y_tr.mean():.0%} / {split.y_te.mean():.0%}")

    # (b)
    knn_raw = make_knn(split.X_tr, 5, "nenhuma").fit(split.X_tr, split.y_tr)
    knn_sc = make_knn(split.X_tr, 5, "padronizada").fit(split.X_tr, split.y_tr)
    acc_raw = accuracy_score(split.y_te, knn_raw.predict(split.X_te))
    acc_sc = accuracy_score(split.y_te, knn_sc.predict(split.X_te))
    part_raw = participacao_na_distancia(split.X_tr, "nenhuma")
    save_table(part_raw.round(6).rename("participacao").to_frame(), "q11b_participacao_distancia")
    print(f"\n(b) k = 5 sem padronização: {acc_raw:.1%} | com padronização: {acc_sc:.1%}")
    print("    Participação na distância euclidiana sem padronização:")
    print((100 * part_raw.head(5)).round(2).to_string())
    plot_padronizacao(acc_raw, acc_sc, part_raw)

    # (c)
    linhas = []
    for k in K_ENUNCIADO:
        m = make_knn(split.X_tr, k).fit(split.X_tr, split.y_tr)
        linhas.append({"k": k, "acc_treino": accuracy_score(split.y_tr, m.predict(split.X_tr)),
                       "acc_teste": accuracy_score(split.y_te, m.predict(split.X_te))})
    curva = pd.DataFrame(linhas).set_index("k")
    save_table(curva.round(4), "q11c_acuracia_por_k")
    melhor = int(curva.index[curva["acc_teste"] == curva["acc_teste"].max()].min())
    print(f"\n(c) Acurácia por k:")
    print((100 * curva).round(1).T.to_string())
    print(f"    Maior acurácia no teste: k = {melhor} ({curva.loc[melhor, 'acc_teste']:.1%})")
    plot_curva_k(curva, melhor)

    # (d)
    final = make_knn(split.X_tr, melhor).fit(split.X_tr, split.y_tr)
    pred = final.predict(split.X_te)
    cm = confusion_matrix(split.y_te, pred)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(split.y_te, pred)
    print(f"\n(d) k = {melhor}: matriz [[VN, FP], [FN, VP]] = {cm.tolist()}")
    print(f"    Maus aprovados (FN) = {fn} de {fn + tp} ({fn / (fn + tp):.0%}) | "
          f"bons recusados (FP) = {fp} de {fp + tn} ({fp / (fp + tn):.0%})")
    plot_matriz(cm, melhor, acc)

    # (e)
    print(f"\n(e) k = 1: treino = {curva.loc[1, 'acc_treino']:.1%} | teste = {curva.loc[1, 'acc_teste']:.1%}")
    print(f"    Linhas duplicadas no treino: {split.X_tr.duplicated().sum()}")

    # (f) Comparação com as listas anteriores, recalculada na mesma divisão
    modelos = {
        f"KNN (k = {melhor})": final,
        "Árvore profundidade 2 (Lista 4, notebook)": make_tree(split.X_tr, 2),
        "Árvore profundidade 4 (Lista 4, após remover COD_CLI)": make_tree(split.X_tr, 4),
        "Regressão logística (Lista 5)": make_logit(split.X_tr),
    }
    comp = pd.DataFrame({n: avaliar(m.fit(split.X_tr, split.y_tr), split)
                         for n, m in modelos.items()}).T
    save_table(comp.round(4), "q11f_comparacao")
    print("\n(f) Comparação na mesma divisão treino/teste:")
    print(comp.round(3).to_string())

    return {"df": df, "split": split, "curva": curva, "melhor": melhor, "comp": comp}
