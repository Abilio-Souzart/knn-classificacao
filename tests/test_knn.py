"""Testes do pré-processamento e de propriedades do KNN usadas na análise."""
import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from src.data import features_e_alvo, load_pejota, make_split
from src.preprocessing import make_knn, signed_log, variaveis_selecionadas


@pytest.fixture(scope="module")
def split():
    return make_split(load_pejota())


def test_identificador_fora(split):
    assert "COD_CLI" not in split.X_tr.columns


def test_padronizacao_so_com_treino(split):
    pipe = make_knn(split.X_tr, 5).fit(split.X_tr, split.y_tr)
    Z_tr = pipe.named_steps["prep"].transform(split.X_tr)
    assert np.allclose(Z_tr.mean(axis=0), 0, atol=1e-8)
    Z_te = pipe.named_steps["prep"].transform(split.X_te)
    assert not np.allclose(Z_te.mean(axis=0), 0, atol=1e-3)


def test_k1_memoriza_treino(split):
    """Sem linhas duplicadas, o vizinho mais próximo de cada ponto é ele mesmo."""
    assert split.X_tr.duplicated().sum() == 0
    pipe = make_knn(split.X_tr, 1).fit(split.X_tr, split.y_tr)
    assert accuracy_score(split.y_tr, pipe.predict(split.X_tr)) == 1.0


def test_selecao_dentro_do_pipeline(split):
    pipe = make_knn(split.X_tr, 25, "log", n_variaveis=2).fit(split.X_tr, split.y_tr)
    assert len(variaveis_selecionadas(pipe)) == 2


def test_signed_log_preserva_ordem():
    x = np.array([-50.0, -1.0, 0.0, 2.0, 1e5])
    assert np.all(np.diff(signed_log(x)) > 0)
