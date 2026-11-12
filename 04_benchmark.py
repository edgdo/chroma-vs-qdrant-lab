"""Passo 4: benchmark Chroma x Qdrant com os MESMOS vetores, parâmetros e cliente.

Por rodada e por banco:
  1. recria a coleção (começa do zero)
  2. ingestão em lotes, cronometrada; espera o índice ficar pronto
  3. aquecimento + 1000 consultas: latência p50/p95/p99 e recall@k (ef padrão)
  4. varredura de ef_search: curva recall x latência
  5. busca com filtro de categoria: latência e recall
  6. QPS com 1, 4, 8, 16 threads
  7. disco usado na VM
`docker stats` das duas VMs fica gravando durante tudo (evidencias/docker/).

Uso: python 04_benchmark.py [--bancos chroma qdrant] [--rodadas 3]
"""
import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

from bancos import BANCOS
from comum import (AQUECIMENTO, ARQ_CHUNKS, ARQ_GABARITO, ARQ_GABARITO_FILTRO, ARQ_VETORES,
                   EF_PADRAO, EF_VARREDURA, K, LOTE_INGESTAO, RES, RODADAS, THREADS_QPS, VMS,
                   MonitorDocker, agora, estado_outras_vms, evento, iniciar_log, ssh)


EXECUCAO = datetime.now().strftime("%Y%m%d-%H%M%S")


def gravar(nome: str, linha: dict) -> None:
    arq = RES / f"{nome}.csv"
    if arq.exists():
        with open(arq, encoding="utf-8") as f:
            cabecalho = f.readline().strip().split(",")
        if cabecalho != list(linha):  # CSV de versão anterior (outras colunas): guarda à parte
            arq.rename(RES / f"{nome}_anterior_{EXECUCAO}.csv")
    novo = not arq.exists()
    with open(arq, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linha))
        if novo:
            w.writeheader()
        w.writerow(linha)


def recall(resultados: list[list[int]], gabarito: np.ndarray) -> float:
    return float(np.mean([len(set(r) & set(g)) / K for r, g in zip(resultados, gabarito)]))


def medir_consultas(db, consultas, cats=None):
    lat, res = [], []
    for i, q in enumerate(consultas):
        t = time.perf_counter()
        ids = db.buscar(q, K, cats[i] if cats is not None else None)
        lat.append((time.perf_counter() - t) * 1000)
        res.append(ids)
    return np.array(lat), res


def resumo_lat(lat: np.ndarray) -> dict:
    return {"p50_ms": round(float(np.percentile(lat, 50)), 3),
            "p95_ms": round(float(np.percentile(lat, 95)), 3),
            "p99_ms": round(float(np.percentile(lat, 99)), 3),
            "media_ms": round(float(lat.mean()), 3)}


def medir_qps(db, consultas, threads: int, repeticoes: int = 3) -> float:
    carga = np.concatenate([consultas] * repeticoes)
    t = time.perf_counter()
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(lambda q: db.buscar(q, K), carga))
    return len(carga) / (time.perf_counter() - t)


def rodar(db, rodada: int, base, ids, metas, textos, consultas, cats_q, gab, gab_f) -> None:
    nome = db.nome
    tag = f"{nome}_r{rodada}"
    # "execucao" identifica cada vez que o 04 roda: os CSVs acumulam execuções e o 06 usa a última
    comum = {"execucao": EXECUCAO, "banco": nome, "rodada": rodada, "n": len(ids)}

    # 1-2. ingestão
    db.recriar()
    evento("04", f"ingestao_{tag}", "inicio")
    t0 = time.perf_counter()
    for i in range(0, len(ids), LOTE_INGESTAO):
        s = slice(i, i + LOTE_INGESTAO)
        db.inserir(ids[s], base[s], metas[s], textos[s])
        if (i // LOTE_INGESTAO) % 40 == 0:
            print(f"  [{nome}] {i + LOTE_INGESTAO}/{len(ids)} "
                  f"({time.perf_counter() - t0:.0f} s)")
    t_insert = time.perf_counter() - t0
    info = db.aguardar_indice()
    t_total = time.perf_counter() - t0
    evento("04", f"ingestao_{tag}", "fim", f"insert_s={t_insert:.1f} total_s={t_total:.1f} {info}")
    assert db.contar() == len(ids), f"{nome}: contagem {db.contar()} != {len(ids)}"
    gravar("ingestao", {**comum, "insert_s": round(t_insert, 2),
                        "indice_pronto_s": round(t_total, 2),
                        "vetores_por_s": round(len(ids) / t_total, 1), "info": str(info)})

    # 3. latência + recall no ef padrão
    evento("04", f"latencia_{tag}", "inicio")
    db.set_ef(EF_PADRAO)
    medir_consultas(db, consultas[:AQUECIMENTO])
    lat, res = medir_consultas(db, consultas)
    r = recall(res, gab)
    gravar("latencia", {**comum, "cenario": "sem_filtro", "ef": EF_PADRAO, **resumo_lat(lat),
                        "recall": round(r, 4)})
    print(f"  [{nome}] ef={EF_PADRAO}: {resumo_lat(lat)} recall@{K}={r:.4f}")
    np.save(RES / f"lat_bruta_{tag}.npy", lat)

    # 4. varredura de ef
    for ef in EF_VARREDURA:
        db.set_ef(ef)
        medir_consultas(db, consultas[:AQUECIMENTO])
        lat, res = medir_consultas(db, consultas)
        r = recall(res, gab)
        gravar("varredura_ef", {**comum, "ef": ef, **resumo_lat(lat), "recall": round(r, 4)})
        print(f"  [{nome}] ef={ef}: p95={np.percentile(lat, 95):.2f} ms recall={r:.4f}")
    db.set_ef(EF_PADRAO)

    # 5. com filtro
    lat, res = medir_consultas(db, consultas, cats_q)
    r = recall(res, gab_f)
    gravar("latencia", {**comum, "cenario": "com_filtro", "ef": EF_PADRAO, **resumo_lat(lat),
                        "recall": round(r, 4)})
    print(f"  [{nome}] filtro: {resumo_lat(lat)} recall@{K}={r:.4f}")
    evento("04", f"latencia_{tag}", "fim")

    # 6. QPS
    evento("04", f"qps_{tag}", "inicio")
    for th in THREADS_QPS:
        qps = medir_qps(db, consultas, th)
        gravar("qps", {**comum, "threads": th, "qps": round(qps, 1)})
        print(f"  [{nome}] {th} threads: {qps:.0f} QPS")
    evento("04", f"qps_{tag}", "fim")

    # 7. disco
    ip, _, pasta = VMS[nome]
    du = ssh(ip, f"sudo du -sb {pasta} | cut -f1")
    gravar("disco", {**comum, "bytes": du, "mb": round(int(du) / 1e6, 1) if du.isdigit() else du})
    print(f"  [{nome}] disco: {du} bytes")


def so_varredura(bancos: dict, rodadas: int, n: int, consultas, gab) -> None:
    """Refaz só a curva recall x latência, usando a coleção que já está carregada nos bancos."""
    for rodada in range(1, rodadas + 1):
        for nome, db in bancos.items():
            db.abrir()
            total = db.contar()
            assert total == n, f"{nome}: coleção tem {total} pontos, esperado {n}; rode o 04 completo"
            print(f"\n===== varredura {rodada}/{rodadas} - {nome} =====")
            evento("04", f"varredura_{nome}_r{rodada}", "inicio")
            for ef in EF_VARREDURA:
                db.set_ef(ef)
                medir_consultas(db, consultas[:AQUECIMENTO])
                lat, res = medir_consultas(db, consultas)
                r = recall(res, gab)
                gravar("varredura_ef", {"execucao": EXECUCAO, "banco": nome, "rodada": rodada, "n": n,
                                        "ef": ef, **resumo_lat(lat), "recall": round(r, 4)})
                print(f"  [{nome}] ef={ef}: p95={np.percentile(lat, 95):.2f} ms recall={r:.4f}")
            db.set_ef(EF_PADRAO)
            evento("04", f"varredura_{nome}_r{rodada}", "fim")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bancos", nargs="+", default=list(BANCOS), choices=list(BANCOS))
    ap.add_argument("--rodadas", type=int, default=RODADAS)
    ap.add_argument("--so-varredura", action="store_true",
                    help="refaz só a varredura de ef_search nas coleções já carregadas")
    a = ap.parse_args()
    iniciar_log("04_benchmark")

    df = pd.read_parquet(ARQ_CHUNKS)
    v = np.load(ARQ_VETORES)
    m = (df.papel == "corpus").to_numpy()
    base, ids = v[m], df.id.to_numpy()[m]
    metas = [{"cat": c, "titulo": t} for c, t in zip(df.cat[m], df.titulo[m])]
    textos = df.texto[m].tolist()
    consultas, cats_q = v[~m], df.cat.to_numpy()[~m]
    gab, gab_f = np.load(ARQ_GABARITO), np.load(ARQ_GABARITO_FILTRO)
    print(f"[{agora()}] corpus={len(ids)} consultas={len(consultas)} bancos={a.bancos} "
          f"rodadas={a.rodadas}")

    vizinhas = estado_outras_vms()
    evento("04", "outras_vms_no_host", "estado", str(vizinhas))
    if any(v == "ligada" for v in vizinhas.values()):
        print("  ATENÇÃO: há outras VMs ligadas no mesmo host; elas disputam CPU com o teste.")

    bancos = {n: BANCOS[n]() for n in a.bancos}
    for n, db in bancos.items():
        print(f"  {n}: versão {db.versao()}")

    monitores = [MonitorDocker(n, "varredura" if a.so_varredura else "benchmark")
                 for n in a.bancos]
    for mon in monitores:
        mon.__enter__()
    try:
        if a.so_varredura:
            so_varredura(bancos, a.rodadas, len(ids), consultas, gab)
            return
        for rodada in range(1, a.rodadas + 1):
            for n, db in bancos.items():  # alterna os bancos dentro de cada rodada
                print(f"\n===== rodada {rodada}/{a.rodadas} - {n} =====")
                rodar(db, rodada, base, ids, metas, textos, consultas, cats_q, gab, gab_f)
                time.sleep(10)  # respiro para separar as fases no docker stats
    finally:
        for mon in monitores:
            mon.__exit__(None, None, None)
    print(f"\n[{agora()}] fim. CSVs em {RES}")


if __name__ == "__main__":
    main()
