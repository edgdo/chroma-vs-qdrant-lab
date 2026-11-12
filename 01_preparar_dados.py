"""Passo 1: baixa a Wikipédia em português (Hugging Face, em streaming) e quebra em chunks.

Saída: dados/chunks.parquet com as colunas
  id, papel ("corpus" ou "consulta"), titulo, url, cat, texto

- Junta parágrafos de pelo menos 200 caracteres em chunks de até ~1000 caracteres.
- Embaralha com semente fixa e separa N_CONSULTAS chunks como consultas (não entram nos bancos).
- `cat` é uma categoria sintética (hash do título, N_CATS valores) para o teste de busca com filtro.
"""
import hashlib
import random
import time

import pandas as pd
from datasets import load_dataset

from comum import ARQ_CHUNKS, N_CATS, N_CONSULTAS, N_CORPUS, SEED, evento, iniciar_log

TAM_MAX = 1000
TAM_MIN_PARAGRAFO = 200


def categoria(titulo: str) -> str:
    return f"cat_{int(hashlib.md5(titulo.encode()).hexdigest(), 16) % N_CATS}"


def main() -> None:
    iniciar_log("01_preparar_dados")
    alvo = N_CORPUS + N_CONSULTAS
    evento("01", "download_chunking", "inicio", f"alvo={alvo}")
    t0 = time.perf_counter()

    ds = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
    linhas, artigos = [], 0
    for art in ds:
        artigos += 1
        titulo, url = art["title"], art["url"]
        paragrafos = [p.strip() for p in art["text"].split("\n")
                      if len(p.strip()) >= TAM_MIN_PARAGRAFO]
        buf = ""
        for p in paragrafos:
            if buf and len(buf) + len(p) + 1 > TAM_MAX:
                linhas.append((titulo, url, buf))
                buf = ""
            buf = f"{buf}\n{p}".strip() if buf else p[: 2 * TAM_MAX]
        if buf:
            linhas.append((titulo, url, buf))
        if artigos % 2000 == 0:
            print(f"  {artigos} artigos lidos, {len(linhas)} chunks")
        if len(linhas) >= alvo:
            break

    linhas = linhas[:alvo]
    random.Random(SEED).shuffle(linhas)
    df = pd.DataFrame(linhas, columns=["titulo", "url", "texto"])
    df["cat"] = df["titulo"].map(categoria)
    df["papel"] = ["consulta"] * N_CONSULTAS + ["corpus"] * (len(df) - N_CONSULTAS)
    # ids: corpus 0..N-1 (são os ids usados nos dois bancos); consultas continuam a numeração
    df = pd.concat([df[df.papel == "corpus"], df[df.papel == "consulta"]], ignore_index=True)
    df.insert(0, "id", range(len(df)))
    df.to_parquet(ARQ_CHUNKS, index=False)

    dt = time.perf_counter() - t0
    evento("01", "download_chunking", "fim", f"chunks={len(df)} artigos={artigos} s={dt:.1f}")
    print(f"\nArtigos lidos: {artigos}")
    print(f"Chunks: {len(df)} (corpus={int((df.papel == 'corpus').sum())}, "
          f"consultas={int((df.papel == 'consulta').sum())})")
    print(f"Tamanho médio do chunk: {df.texto.str.len().mean():.0f} caracteres")
    print("Distribuição por categoria:\n" + df.cat.value_counts().sort_index().to_string())
    print(f"Salvo em {ARQ_CHUNKS} ({dt:.1f} s)")


if __name__ == "__main__":
    main()
