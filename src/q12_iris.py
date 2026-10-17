"""Questão 12: KNN na base Iris (carregada do próprio scikit-learn, sem arquivo)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import K_IRIS, SEED, TEST_SIZE_IRIS
from .utils import header, save_figure, save_table


def knn(k):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))


def run() -> dict:
    header("QUESTÃO 12 - KNN na base Iris")
    iris = load_iris(as_frame=True)
    X, y, nomes = iris.data, iris.target, list(iris.target_names)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE_IRIS,
                                              random_state=SEED, stratify=y)
    print(f"Treino: {len(y_tr)} | Teste: {len(y_te)} (cada flor = {1 / len(y_te):.1%} de acurácia)")

    # (a)
    accs = {k: accuracy_score(y_te, knn(k).fit(X_tr, y_tr).predict(X_te)) for k in K_IRIS}
    erros = {k: int(round((1 - a) * len(y_te))) for k, a in accs.items()}
    melhor = max(accs, key=accs.get)
    tab = pd.DataFrame({"acurácia teste": accs, "flores erradas": erros})
    save_table(tab.round(4), "q12a_acuracia_k")
    print("(a)"); print(tab.round(4).to_string())
    print(f"    Melhor k: {melhor}")

    # (b)
    pred = knn(melhor).fit(X_tr, y_tr).predict(X_te)
    cm = confusion_matrix(y_te, pred)
    print(f"\n(b) Matriz de confusão (k = {melhor}):")
    print(pd.DataFrame(cm, index=nomes, columns=nomes).to_string())
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=nomes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"(b) Iris: KNN k = {melhor}, acurácia = {accs[melhor]:.1%}")
    fig.tight_layout()
    save_figure(fig, "q12b_matriz_confusao")

    # Complemento: validação cruzada repetida
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    ks = list(range(1, 31))
    res = pd.DataFrame({k: cross_val_score(knn(k), X, y, cv=cv) for k in ks})
    resumo = pd.DataFrame({"média": res.mean(), "desvio entre folds": res.std()})
    save_table(resumo.round(4), "q12_cv_repetida")
    print("\nComplemento: CV 5 folds x 20 repetições (base completa):")
    print(resumo.loc[K_IRIS].round(3).to_string())
    print(f"    Faixa das médias para k = 1..30: {resumo['média'].min():.3f} a {resumo['média'].max():.3f}")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ks, resumo["média"], "o-", color="#1565C0", label="Média CV (5 folds x 20)")
    ax.fill_between(ks, resumo["média"] - resumo["desvio entre folds"],
                    resumo["média"] + resumo["desvio entre folds"], alpha=0.15, color="#1565C0",
                    label="± 1 desvio entre folds")
    for k in K_IRIS:
        ax.plot(k, accs[k], "s", color="#F44336", ms=8)
    ax.plot([], [], "s", color="#F44336", label="Teste único (38 flores)")
    ax.set_xlabel("k")
    ax.set_ylabel("Acurácia")
    ax.set_ylim(0.85, 1.0)
    ax.set_title("Iris: no teste único, a diferença entre k = 1, 5 e 15 é de 1 a 2 flores")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save_figure(fig, "q12_cv_repetida")

    # Sobreposição versicolor x virginica
    sobrepoe = {}
    for col in X.columns:
        a, b = X.loc[y == 1, col], X.loc[y == 2, col]
        sobrepoe[col] = float(((a >= b.min()) & (a <= b.max())).mean())
    print("    % de versicolor dentro da faixa de virginica, por medida:",
          {k: f"{v:.0%}" for k, v in sobrepoe.items()})
    return {"accs": accs, "cm": cm, "resumo": resumo}
