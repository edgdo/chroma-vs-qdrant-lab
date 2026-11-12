"""Passo 2: gera os embeddings de todos os chunks com bge-m3 no Ollama (GPU), com evidências.

- Monitor de GPU (nvidia-smi a cada 1 s) ligado durante todo o passo.
- Salva `ollama ps` antes/durante (mostra o modelo carregado e % na GPU).
- Grava em partes (dados/emb_partes/) e retoma de onde parou se for interrompido.
Saída: dados/vetores.npy (float32, normalizado, mesma ordem de dados/chunks.parquet).
"""
import subprocess
import time

import numpy as np
import pandas as pd
import requests

from comum import (ARQ_CHUNKS, ARQ_VETORES, DADOS, DIM, EMBED_MODEL, EVID, LOTE_EMBED, OLLAMA_URL,
                   MonitorGPU, agora, evento, iniciar_log)

PARTE = 5000  # chunks por arquivo parcial


def embed(textos: list[str]) -> np.ndarray:
    r = requests.post(f"{OLLAMA_URL}/api/embed",
                      json={"model": EMBED_MODEL, "input": textos, "truncate": True,
                            "keep_alive": "30m"},
                      timeout=600)
    r.raise_for_status()
    v = np.asarray(r.json()["embeddings"], dtype=np.float32)
    assert v.shape == (len(textos), DIM), v.shape
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def salvar_ollama_ps(rotulo: str) -> None:
    saida = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    with open(EVID / "ollama_ps.txt", "a", encoding="utf-8") as f:
        f.write(f"===== {agora()} ({rotulo})\n{saida}\n")
    print(f"ollama ps ({rotulo}):\n{saida}")


def main() -> None:
    iniciar_log("02_gerar_embeddings")
    df = pd.read_parquet(ARQ_CHUNKS)
    textos = df.texto.tolist()
    partes_dir = DADOS / "emb_partes"
    partes_dir.mkdir(exist_ok=True)
    n_partes = (len(textos) + PARTE - 1) // PARTE

    print(f"{len(textos)} chunks, modelo {EMBED_MODEL}, lote {LOTE_EMBED}, {n_partes} partes")
    embed(["aquecimento"])  # carrega o modelo na GPU antes de medir
    salvar_ollama_ps("modelo carregado")

    evento("02", "embeddings", "inicio", f"n={len(textos)} modelo={EMBED_MODEL}")
    t0 = time.perf_counter()
    feitos = 0
    with MonitorGPU("embeddings"):
        for i in range(n_partes):
            arq = partes_dir / f"parte_{i:04d}.npy"
            ini, fim = i * PARTE, min((i + 1) * PARTE, len(textos))
            if arq.exists():
                print(f"  parte {i + 1}/{n_partes} já existe, pulando")
                continue
            tp = time.perf_counter()
            vetores = np.concatenate([embed(textos[j:min(j + LOTE_EMBED, fim)])
                                      for j in range(ini, fim, LOTE_EMBED)])
            assert vetores.shape[0] == fim - ini, vetores.shape
            np.save(arq, vetores)
            feitos += fim - ini
            dt = time.perf_counter() - tp
            total = time.perf_counter() - t0
            print(f"  parte {i + 1}/{n_partes}: {fim - ini} chunks em {dt:.1f} s "
                  f"({(fim - ini) / dt:.0f}/s) | acumulado {feitos} em {total:.0f} s")
            if i == 0:
                salvar_ollama_ps("durante a geração")
    dt = time.perf_counter() - t0
    evento("02", "embeddings", "fim", f"gerados={feitos} s={dt:.1f}")

    # Cada parte i corresponde a textos[i*PARTE : (i+1)*PARTE]. O corte [:n] também corrige
    # partes geradas pela versão anterior, que gravava 56 vetores a mais no fim de cada parte
    # (o último lote de 64 avançava sobre a parte seguinte; as primeiras PARTE linhas estão certas).
    blocos = []
    for i in range(n_partes):
        n = min((i + 1) * PARTE, len(textos)) - i * PARTE
        p = np.load(partes_dir / f"parte_{i:04d}.npy")
        assert p.shape[0] >= n, (i, p.shape)
        blocos.append(p[:n])
    todos = np.concatenate(blocos)
    assert todos.shape == (len(textos), DIM)
    np.save(ARQ_VETORES, todos)
    if feitos:
        print(f"\nGerados nesta execução: {feitos} embeddings em {dt:.1f} s "
              f"({feitos / dt:.0f} chunks/s)")
    print(f"Salvo em {ARQ_VETORES}: {todos.shape}, {todos.nbytes / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
