"""Passo 5: demonstração de RAG ponta a ponta, com a mesma pergunta nos dois bancos.

pergunta -> bge-m3 (GPU) -> top-5 no Chroma e no Qdrant -> qwen2.5:14b (GPU) responde citando.
Grava evidencias/rag_demo_<ts>.md com trechos recuperados, respostas e tempos de cada etapa,
com o monitor de GPU ligado. Precisa que o 04_benchmark já tenha carregado as coleções.

Uso: python 05_rag_demo.py ["pergunta 1" "pergunta 2" ...]
     python 05_rag_demo.py -i      (modo interativo: digita as perguntas uma a uma)
Pode ser rodado a qualquer momento depois do 04, enquanto as coleções estiverem nos bancos.
"""
import sys
import time
from datetime import datetime

import numpy as np
import requests

from bancos import BANCOS
from comum import (EMBED_MODEL, EVID, LLM_MODEL, OLLAMA_URL, MonitorGPU, agora, evento,
                   iniciar_log)

PERGUNTAS_PADRAO = [
    "Qual é a capital do Brasil e quando ela foi inaugurada?",
    "O que foi a Guerra das Malvinas e quem venceu?",
    "Como funciona a fotossíntese?",
]
TOP = 5


def embed(texto: str) -> np.ndarray:
    r = requests.post(f"{OLLAMA_URL}/api/embed",
                      json={"model": EMBED_MODEL, "input": [texto]}, timeout=120)
    r.raise_for_status()
    v = np.asarray(r.json()["embeddings"][0], dtype=np.float32)
    return v / np.linalg.norm(v)


def gerar(pergunta: str, trechos: list[dict]) -> dict:
    contexto = "\n\n".join(f"[{i + 1}] ({t['titulo']}) {t['texto']}" for i, t in enumerate(trechos))
    prompt = (
        "Responda à pergunta usando SOMENTE os trechos abaixo. Cite os trechos usados como [n]. "
        "Se os trechos não tiverem a resposta, diga que não encontrou.\n\n"
        f"Trechos:\n{contexto}\n\nPergunta: {pergunta}\nResposta:"
    )
    r = requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": LLM_MODEL, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0}, "keep_alive": "10m"},
                      timeout=600)
    r.raise_for_status()
    d = r.json()
    tok_s = d["eval_count"] / (d["eval_duration"] / 1e9) if d.get("eval_duration") else 0
    return {"resposta": d["response"].strip(), "tokens": d.get("eval_count"),
            "tokens_por_s": round(tok_s, 1), "total_s": round(d["total_duration"] / 1e9, 2)}


def main() -> None:
    iniciar_log("05_rag_demo")
    interativo = sys.argv[1:] == ["-i"]
    if interativo:
        def ler():
            print("Modo interativo: digite a pergunta e Enter (Enter vazio encerra e salva).")
            while True:
                try:
                    p = input("\npergunta> ").strip()
                except EOFError:
                    return
                if not p:
                    return
                yield p
        perguntas = ler()
    else:
        perguntas = sys.argv[1:] or PERGUNTAS_PADRAO
    bancos = {n: cls() for n, cls in BANCOS.items()}
    for db in bancos.values():
        db.abrir()

    md = [f"# Demonstração RAG: Chroma x Qdrant\n\nGerado em {agora()}  ",
          f"Embedding: `{EMBED_MODEL}` | LLM: `{LLM_MODEL}` | top-{TOP}\n"]
    evento("05", "rag", "inicio", "interativo" if interativo else f"perguntas={len(perguntas)}")
    with MonitorGPU("rag"):
        for p in perguntas:
            print(f"\n=== {p}")
            md.append(f"\n## {p}\n")
            t = time.perf_counter()
            v = embed(p)
            t_emb = (time.perf_counter() - t) * 1000
            md.append(f"Embedding da pergunta: {t_emb:.1f} ms\n")
            for nome, db in bancos.items():
                t = time.perf_counter()
                trechos = db.buscar_com_texto(v, TOP)
                t_busca = (time.perf_counter() - t) * 1000
                g = gerar(p, trechos)
                print(f"[{nome}] {p} | busca {t_busca:.1f} ms | LLM {g['total_s']} s "
                      f"({g['tokens_por_s']} tok/s)")
                md.append(f"\n### {nome}\n\nBusca: {t_busca:.1f} ms | LLM: {g['total_s']} s, "
                          f"{g['tokens']} tokens, {g['tokens_por_s']} tokens/s\n")
                md.append("Trechos recuperados:\n")
                for i, tr in enumerate(trechos):
                    md.append(f"{i + 1}. **{tr['titulo']}** (id {tr['id']}, score "
                              f"{tr['score']:.3f}): {tr['texto'][:200]}...")
                md.append(f"\n**Resposta:**\n\n{g['resposta']}\n")
                print("  trechos: " + " | ".join(f"[{i + 1}] {tr['titulo']} ({tr['score']:.2f})"
                                                for i, tr in enumerate(trechos)))
                print(f"  resposta ({nome}): {g['resposta']}\n")
    evento("05", "rag", "fim")

    saida = EVID / f"rag_demo_{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    saida.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSalvo em {saida}")


if __name__ == "__main__":
    main()
