# Autor: Alessandro Stefani
# Disciplina: Tech Challenge — Fase 01 (Produtização de Modelos)
# Descrição: Configuração de logging estruturado (substitui print() em todo o pacote).
"""Setup único de logging para módulos e API (sem `print()`).

Formato com campos fixos (timestamp | nível | logger | mensagem); os logs de evento
da API usam pares `chave=valor` (método, rota, status, latência) para serem fáceis
de parsear/agregar.
"""
from __future__ import annotations

import logging

from . import config

_FORMATO = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configurado = False


def configurar_logging(nivel: str | None = None) -> None:
    """Configura o root logger uma única vez (idempotente)."""
    global _configurado
    if _configurado:
        return
    logging.basicConfig(level=(nivel or config.LOG_LEVEL), format=_FORMATO)
    _configurado = True
