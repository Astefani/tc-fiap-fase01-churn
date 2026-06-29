# Autor: Alessandro Stefani
# Disciplina: Tech Challenge — Fase 01 (Produtização de Modelos)
# Descrição: MLP (PyTorch) + adaptador sklearn para entrar no Pipeline serializável.
"""Rede neural (MLP) e seu wrapper compatível com scikit-learn.

`TorchMLPClassifier` faz a MLP se comportar como um estimador sklearn: assim ela
entra no `Pipeline` (FeatureEngineer → ColumnTransformer → modelo) e o `cross_validate`
funciona com ela direto — cada `fit` faz um sub-split interno para o early stopping.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from . import config


def escolher_device() -> torch.device:
    """CUDA (Ubuntu/RTX) → MPS (Mac) → CPU. Treino usa o melhor disponível."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLP(nn.Module):
    """Blocos Linear → ReLU → Dropout; saída em 1 logit (para BCEWithLogits)."""

    def __init__(self, in_dim: int, hidden, dropout: float):
        super().__init__()
        camadas, d = [], in_dim
        for h in hidden:
            camadas += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        camadas += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*camadas)

    def forward(self, x):
        return self.net(x)


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """MLP PyTorch como classificador sklearn (early stopping em val PR-AUC).

    Espera `X` já pré-processado (numérico). No `fit` faz um sub-split estratificado
    interno (`val_size`) para o early stopping e calcula `pos_weight` do sub-treino
    (desbalanceamento). Após o fit, move o modelo para CPU → artefato portável
    (treina na GPU, serve na CPU).
    """

    def __init__(self, hidden=(64, 32), dropout=0.3, lr=1e-3, batch=64,
                 max_epochs=200, patience=20, val_size=0.2, seed=config.SEED):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.batch = batch
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_size = val_size
        self.seed = seed

    def _semear(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def fit(self, X, y):
        self._semear()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.float32).ravel()
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        device = escolher_device()

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.val_size, stratify=y, random_state=self.seed)

        n_pos = max(int((y_tr == 1).sum()), 1)
        pos_weight = torch.tensor([float((y_tr == 0).sum()) / n_pos], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        modelo = MLP(self.n_features_in_, list(self.hidden), self.dropout).to(device)
        opt = torch.optim.Adam(modelo.parameters(), lr=self.lr)

        Xtr_t = torch.tensor(X_tr, device=device)
        ytr_t = torch.tensor(y_tr, device=device).unsqueeze(1)
        Xval_t = torch.tensor(X_val, device=device)
        gerador = torch.Generator().manual_seed(self.seed)   # shuffle determinístico
        loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=self.batch,
                            shuffle=True, generator=gerador)

        melhor = {"prauc": -1.0, "state": None, "epoca": 0}
        espera, historico = 0, []
        for epoca in range(1, self.max_epochs + 1):
            modelo.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(modelo(xb), yb)
                loss.backward()
                opt.step()
            modelo.eval()
            with torch.no_grad():
                p_val = torch.sigmoid(modelo(Xval_t)).cpu().numpy().ravel()
            vprauc = float(average_precision_score(y_val, p_val))
            historico.append((epoca, float(loss.item()), vprauc))
            if vprauc > melhor["prauc"]:
                melhor = {"prauc": vprauc, "state": copy.deepcopy(modelo.state_dict()),
                          "epoca": epoca}
                espera = 0
            else:
                espera += 1
                if espera >= self.patience:
                    break

        modelo.load_state_dict(melhor["state"])
        modelo.to("cpu").eval()             # artefato portável (serving em CPU)
        self.module_ = modelo
        self.best_epoch_ = melhor["epoca"]
        self.history_ = historico
        return self

    def _logits(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            return self.module_(torch.from_numpy(X)).numpy().ravel()

    def decision_function(self, X) -> np.ndarray:
        """Score 1-D (logit) — usado pelos scorers de ranking (PR-AUC/ROC-AUC) na CV."""
        return self._logits(X)

    def predict_proba(self, X) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self._logits(X)))      # sigmoid
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
