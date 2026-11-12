"""Passo 6: gera os gráficos (resultados/*.png) e o resumo resultados/RESULTADOS.md.

Usa a MEDIANA das rodadas. Também plota o uso de GPU (evidencias/gpu/) e de CPU/RAM dos
containers (evidencias/docker/) ao longo do tempo.

Uso: python 06_graficos.py              (usa a execução mais recente do 04)
     python 06_graficos.py 20261110-191500   (usa uma execução específica, coluna "execucao")
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from comum import EVID, RES, agora, iniciar_log  # noqa: E402

# Paleta da apresentação (modelo FIAP MBA): rosa da faixa/chevron e grafite da barra do topo.
CORES = {"chroma": "#ED145B", "qdrant": "#1D1E1F"}  # magenta FIAP e grafite
CINZA = "#8C8C8C"
from matplotlib import font_manager  # noqa: E402

_disponiveis = {f.name for f in font_manager.fontManager.ttflist}
_fonte = next((f for f in ("Arial", "Liberation Sans", "DejaVu Sans") if f in _disponiveis),
              "sans-serif")  # escolhe uma fonte instalada, sem avisos de "findfont"
plt.rcParams.update({
    "font.family": _fonte,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": CINZA, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": "white",
})


def painel(ing: pd.DataFrame, lat: pd.DataFrame, var: pd.DataFrame, q: pd.DataFrame) -> None:
    """Painel 2x2 em 16:9 para a página de Resultados da apresentação."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.75))
    bancos = sorted(ing.banco.unique())
    cores = [CORES[b] for b in bancos]

    t = ing.groupby("banco").vetores_por_s.median().reindex(bancos)
    ax = axs[0, 0]
    barras = ax.bar(bancos, t.values, color=cores, width=0.5)
    ax.bar_label(barras, fmt="%.0f", fontsize=9)
    ax.set_title(f"Ingestão ({int(ing.n.iloc[0]):,} vetores)".replace(",", "."))
    ax.set_ylabel("vetores/s até o índice pronto")

    t = (lat[lat.cenario == "sem_filtro"].groupby("banco")[["p50_ms", "p95_ms", "p99_ms"]]
         .median().reindex(bancos))
    ax = axs[0, 1]
    x = range(3)
    for i, b in enumerate(bancos):
        xs = [v + (i - 0.5) * 0.35 for v in x]
        barras = ax.bar(xs, t.loc[b].values, width=0.35, color=CORES[b], label=b)
        ax.bar_label(barras, fmt="%.1f", fontsize=8)
    ax.set_xticks(list(x), ["p50", "p95", "p99"])
    ax.set_title("Latência por consulta (top-10, ef=100)")
    ax.set_ylabel("ms")
    ax.legend()

    t = var.groupby(["banco", "ef"])[["p95_ms", "recall"]].median().reset_index()
    ax = axs[1, 0]
    for b, g in t.groupby("banco"):
        ax.plot(g.recall, g.p95_ms, "o-", color=CORES[b], label=b)
        for _, r in g.iterrows():
            ax.annotate(f"{int(r.ef)}", (r.recall, r.p95_ms), fontsize=7, color=CINZA,
                        textcoords="offset points", xytext=(4, 4))
    ax.set_title("Recall x latência (rótulo = ef_search)")
    ax.set_xlabel("recall@10")
    ax.set_ylabel("latência p95 (ms)")
    ax.legend()

    t = q.groupby(["banco", "threads"]).qps.median().reset_index()
    ax = axs[1, 1]
    for b, g in t.groupby("banco"):
        ax.plot(g.threads, g.qps, "o-", color=CORES[b], label=b)
    ax.set_xticks(sorted(t.threads.unique()))
    ax.set_title("Throughput com concorrência")
    ax.set_xlabel("threads no cliente")
    ax.set_ylabel("consultas/s")
    ax.legend()
    salvar(fig, "painel_resultados.png")


def grafico_gpu_apresentacao() -> None:
    """GPU durante a geração de embeddings: uso % e potência, 16:9 baixo para a página de Método."""
    # usa o maior arquivo (a execução que de fato gerou os vetores; reexecuções que só
    # remontam as partes criam arquivos de poucos segundos)
    arqs = sorted((EVID / "gpu").glob("*_embeddings.csv"), key=lambda p: p.stat().st_size)
    if not arqs:
        return
    df = pd.read_csv(arqs[-1], skipinitialspace=True)
    df.columns = [c.split(" [")[0].strip() for c in df.columns]
    for c in ("utilization.gpu", "memory.used", "power.draw"):
        df[c] = pd.to_numeric(df[c].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
    df["min"] = [i / 60 for i in range(len(df))]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.fill_between(df["min"], df["utilization.gpu"], color=CORES["chroma"], alpha=0.25)
    ax.plot(df["min"], df["utilization.gpu"], color=CORES["chroma"], label="uso da GPU (%)")
    ax.set_ylim(0, 105)
    ax.set_ylabel("uso (%)")
    ax.set_xlabel("minutos")
    ax2 = ax.twinx()
    ax2.plot(df["min"], df["power.draw"], color=CORES["qdrant"], lw=1, label="potência (W)")
    ax2.set_ylabel("W")
    ax2.grid(False)
    ax.set_title(f"{df['name'].iloc[0]} gerando embeddings bge-m3: média "
                 f"{df['utilization.gpu'].mean():.0f}%, VRAM pico {df['memory.used'].max():.0f} MiB")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")
    salvar(fig, "gpu_apresentacao.png")


def salvar(fig, nome: str) -> None:
    fig.tight_layout()
    fig.savefig(RES / nome, dpi=150)
    plt.close(fig)
    print(f"  {RES / nome}")


def md_tabela(df: pd.DataFrame) -> str:
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    linhas = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([cab, sep, *linhas])


def ler(nome: str, execucao: str | None = None) -> pd.DataFrame:
    """Lê resultados/<nome>.csv só da execução escolhida do 04 (padrão: a mais recente)."""
    df = pd.read_csv(RES / f"{nome}.csv")
    if "execucao" in df.columns:
        df["execucao"] = df["execucao"].astype(str)
        df = df[df.execucao == (execucao or df.execucao.max())]
    return df


def mem_mb(s: str) -> float:
    v = s.split("/")[0].strip()
    for suf, mult in (("GiB", 1024), ("MiB", 1), ("KiB", 1 / 1024), ("B", 1 / 1024 / 1024)):
        if v.endswith(suf):
            return float(v[: -len(suf)]) * mult
    return float("nan")


EXEC_ESCOLHIDA = sys.argv[1] if len(sys.argv) > 1 else None


def main() -> None:
    iniciar_log("06_graficos")
    execs = sorted(pd.read_csv(RES / "ingestao.csv").get("execucao", pd.Series(dtype=str))
                   .dropna().astype(str).unique())
    usada = EXEC_ESCOLHIDA or (execs[-1] if execs else "única")
    print(f"Execuções do 04 encontradas: {execs or ['(sem coluna execucao)']} | usando: {usada}")
    md = [f"# Resultados: Chroma x Qdrant\n\nGerado em {agora()}. Execução do benchmark: "
          f"{usada}. Valores = mediana das rodadas.\n"]
    manif = sorted(EVID.glob("manifesto_*.json"))
    if manif:
        m = json.loads(manif[-1].read_text(encoding="utf-8"))
        md.append("## Ambiente do teste\n")
        md.append("| Item | Valor |\n|---|---|")
        for k, v in m.get("infra", {}).items():
            md.append(f"| {k} | {v} |")
        md.append(f"| GPU do cliente | {m['host'].get('gpu')} |")
        md.append(f"| CPU do cliente | {m['host'].get('cpu')} |")
        for n, vm in m.get("vms", {}).items():
            md.append(f"| VM {n} | {vm.get('so')}, kernel {vm.get('kernel')}, {vm.get('vcpu')} vCPU "
                      f"{vm.get('cpu_modelo')}, mem {vm.get('memoria')}, imagem "
                      f"`{vm.get('imagem', '').split()[0] if vm.get('imagem') else ''}` |")
        md.append(f"| Outras VMs no host (manifesto) | {m.get('outras_vms_no_host')} |")
        md.append(f"| Pacotes Python | {m.get('pacotes_python')} |\n")

    # ingestão
    ing = ler("ingestao", EXEC_ESCOLHIDA)
    t = ing.groupby("banco")[["insert_s", "indice_pronto_s", "vetores_por_s"]].median().round(1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(t.index, t.vetores_por_s, color=[CORES[b] for b in t.index])
    ax.set_ylabel("vetores/s (até o índice pronto)")
    ax.set_title(f"Ingestão de {int(ing.n.iloc[0]):,} vetores".replace(",", "."))
    salvar(fig, "01_ingestao.png")
    md += ["## Ingestão\n", md_tabela(t.reset_index()), ""]

    # latência
    lat = ler("latencia", EXEC_ESCOLHIDA)
    t = (lat.groupby(["banco", "cenario"])[["p50_ms", "p95_ms", "p99_ms", "recall"]]
         .median().round(3).reset_index())
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, cen in zip(axs, ["sem_filtro", "com_filtro"]):
        s = t[t.cenario == cen].set_index("banco")
        s[["p50_ms", "p95_ms", "p99_ms"]].T.plot.bar(ax=ax, color=[CORES[b] for b in s.index],
                                                     rot=0)
        ax.set_title(cen.replace("_", " "))
        ax.set_ylabel("ms")
    salvar(fig, "02_latencia.png")
    md += ["## Latência e recall (ef padrão)\n", md_tabela(t), ""]

    # recall x latência
    var = ler("varredura_ef", EXEC_ESCOLHIDA)
    t = var.groupby(["banco", "ef"])[["p95_ms", "recall"]].median().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    for b, g in t.groupby("banco"):
        ax.plot(g.recall, g.p95_ms, "o-", color=CORES[b], label=b)
        for _, r in g.iterrows():
            ax.annotate(f"ef={int(r.ef)}", (r.recall, r.p95_ms), fontsize=7,
                        textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel(f"recall@10")
    ax.set_ylabel("latência p95 (ms)")
    ax.set_title("Recall x latência (varredura de ef_search)")
    ax.legend()
    salvar(fig, "03_recall_latencia.png")
    md += ["## Varredura de ef_search\n", md_tabela(t.round(4)), ""]

    # QPS
    q = ler("qps", EXEC_ESCOLHIDA)
    t = q.groupby(["banco", "threads"]).qps.median().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    for b, g in t.groupby("banco"):
        ax.plot(g.threads, g.qps, "o-", color=CORES[b], label=b)
    ax.set_xlabel("threads cliente")
    ax.set_ylabel("consultas/s")
    ax.set_title("Throughput")
    ax.legend()
    salvar(fig, "04_qps.png")
    md += ["## Throughput (QPS)\n", md_tabela(t.pivot(index="threads", columns="banco",
                                                      values="qps").reset_index()), ""]

    # figuras prontas para a apresentação
    painel(ing, lat, var, q)
    grafico_gpu_apresentacao()

    # disco
    d = ler("disco", EXEC_ESCOLHIDA)
    d["mb"] = pd.to_numeric(d.mb, errors="coerce")
    md += ["## Disco\n", md_tabela(d.groupby("banco").mb.median().reset_index()), ""]

    # recursos dos containers
    arqs = sorted((EVID / "docker").glob("*_benchmark_*.csv"))
    if arqs:
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
        linhas_md = []
        for arq in arqs[-2:]:
            df = pd.read_csv(arq, sep=";")
            if df.empty:
                continue
            b = df.container.iloc[0]
            df["ts"] = pd.to_datetime(df.timestamp)
            df["cpu"] = df.cpu.str.rstrip("%").astype(float)
            df["mem"] = df.mem_uso.map(mem_mb)
            axs[0].plot(df.ts, df.cpu, color=CORES.get(b), label=b)
            axs[1].plot(df.ts, df.mem, color=CORES.get(b), label=b)
            linhas_md.append({"banco": b, "cpu_pico_%": df.cpu.max(),
                              "ram_pico_MB": round(df.mem.max(), 0)})
        axs[0].set_ylabel("CPU % (200% = 2 vCPU)")
        axs[1].set_ylabel("RAM (MB)")
        for ax in axs:
            ax.legend()
        axs[0].set_title("Recursos dos containers durante o benchmark (cada VM no seu relógio)")
        salvar(fig, "05_recursos_containers.png")
        md += ["## Recursos (pico durante o benchmark)\n", md_tabela(pd.DataFrame(linhas_md)), ""]

    # GPU
    md.append("## GPU (evidência de uso)\n")
    for arq in sorted((EVID / "gpu").glob("*.csv")):
        df = pd.read_csv(arq, skipinitialspace=True)
        df.columns = [c.split(" [")[0].strip() for c in df.columns]
        if df.empty:
            continue
        for c in ("utilization.gpu", "memory.used", "power.draw"):
            df[c] = pd.to_numeric(df[c].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
        df["s"] = range(len(df))
        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axs[0].plot(df.s, df["utilization.gpu"], color="#76B900")
        axs[0].set_ylabel("GPU %")
        axs[1].plot(df.s, df["memory.used"], color="#76B900")
        axs[1].set_ylabel("VRAM (MiB)")
        axs[2].plot(df.s, df["power.draw"], color="#76B900")
        axs[2].set_ylabel("W")
        axs[2].set_xlabel("segundos")
        axs[0].set_title(f"GPU ({df['name'].iloc[0]}) - {Path(arq).stem}")
        salvar(fig, f"gpu_{Path(arq).stem}.png")
        md.append(f"- GPU `{Path(arq).stem}`: {len(df)} s, uso médio "
                  f"{df['utilization.gpu'].mean():.0f}%, pico {df['utilization.gpu'].max():.0f}%, "
                  f"VRAM pico {df['memory.used'].max():.0f} MiB, potência pico "
                  f"{df['power.draw'].max():.0f} W")

    (RES / "RESULTADOS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Resumo em {RES / 'RESULTADOS.md'}")


if __name__ == "__main__":
    main()
