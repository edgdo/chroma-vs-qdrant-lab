"""Passo 3: calcula o gabarito (top-k EXATO, força bruta) de cada consulta.

É a "resposta certa" contra a qual o recall dos bancos é medido.
- gabarito.npy: top-k sem filtro
- gabarito_filtro.npy: top-k só entre chunks da mesma categoria da consulta
Como os vetores são normalizados, similaridade de cosseno = produto escalar.
"""
import time

import numpy as np
import pandas as pd

from comum import (ARQ_CHUNKS, ARQ_GABARITO, ARQ_GABARITO_FILTRO, ARQ_VETORES, K, evento,
                   iniciar_log)


def topk(q: np.ndarray, base: np.ndarray, ids: np.ndarray) -> np.ndarray:
    s = q @ base.T
    idx = np.argpartition(-s, K, axis=1)[:, :K]
    ordem = np.take_along_axis(s, idx, axis=1).argsort(axis=1)[:, ::-1]
    return ids[np.take_along_axis(idx, ordem, axis=1)]


def main() -> None:
    iniciar_log("03_gabarito")
    df = pd.read_parquet(ARQ_CHUNKS)
    v = np.load(ARQ_VETORES)
    corpus_mask = (df.papel == "corpus").to_numpy()
    base, ids = v[corpus_mask], df.id.to_numpy()[corpus_mask]
    consultas = v[~corpus_mask]
    cat_base = df.cat.to_numpy()[corpus_mask]
    cat_q = df.cat.to_numpy()[~corpus_mask]

    evento("03", "gabarito", "inicio", f"consultas={len(consultas)} base={len(base)}")
    t0 = time.perf_counter()
    gab = np.concatenate([topk(consultas[i:i + 100], base, ids)
                          for i in range(0, len(consultas), 100)])
    gab_f = np.zeros_like(gab)
    for c in np.unique(cat_q):
        mq, mb = cat_q == c, cat_base == c
        gab_f[mq] = topk(consultas[mq], base[mb], ids[mb])
    np.save(ARQ_GABARITO, gab)
    np.save(ARQ_GABARITO_FILTRO, gab_f)
    evento("03", "gabarito", "fim", f"s={time.perf_counter() - t0:.1f}")
    print(f"Gabarito: {gab.shape}, gabarito com filtro: {gab_f.shape}")


if __name__ == "__main__":
    main()
