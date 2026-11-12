"""Configuração e utilitários compartilhados pelos scripts do benchmark Chroma x Qdrant.

Tudo que é ajustável fica aqui ou em variável de ambiente (BDA_*), para os scripts
não terem números mágicos espalhados.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------- pastas
BASE = Path(__file__).resolve().parent
# dados/ é pesado (~1 GB com os vetores e as partes): BDA_DADOS=/outro/caminho manda essa
# pasta para outro disco. Evidências e resultados ficam aqui.
DADOS = Path(os.environ["BDA_DADOS"]).expanduser() if os.environ.get("BDA_DADOS") else BASE / "dados"
EVID = BASE / "evidencias"
RES = BASE / "resultados"
for _d in (DADOS, EVID, EVID / "logs", EVID / "gpu", EVID / "docker", RES):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- infraestrutura
CHROMA_HOST = os.environ.get("BDA_CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("BDA_CHROMA_PORT", "8000"))
QDRANT_URL = os.environ.get("BDA_QDRANT_URL", "http://localhost:6333")
VMS = {  # nome lógico -> (ip, container, pasta de dados na VM)
    "chroma": (CHROMA_HOST, "chroma", "/root/chroma-data"),
    "qdrant": (QDRANT_URL.split("//")[1].split(":")[0], "qdrant", "/root/qdrant-data"),
}
SSH_USER = os.environ.get("BDA_SSH_USER", "labuser")
SSH_KEY = os.path.expanduser(os.environ.get("BDA_SSH_KEY", "~/.ssh/lab"))
OLLAMA_URL = os.environ.get("BDA_OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("BDA_EMBED_MODEL", "bge-m3")
LLM_MODEL = os.environ.get("BDA_LLM_MODEL", "qwen2.5:14b")

# ---------------------------------------------------------------- descrição da infra do teste
# Entra no manifesto e no RESULTADOS.md. Descreva o ambiente do seu teste antes da rodada
# valendo (no host Proxmox: pveversion; free -h; pvesm status).
INFRA = {
    "hipervisor": "PREENCHER (ex.: Proxmox VE x.y, nó <nome>)",
    "host_cpu": "PREENCHER (modelo, núcleos/threads)",
    "host_memoria": "PREENCHER (free -h no host)",
    "host_storage": "PREENCHER (tipo e disco: pvesm status; SSD/NVMe/HDD, ZFS/LVM)",
    "provisionamento": "PREENCHER (ex.: Terraform, clone de template Ubuntu 24.04 cloud-init)",
    "vms": "2 VMs idênticas (bda-chroma, bda-qdrant): 2 vCPU tipo host, 4 GiB RAM dedicada, "
           "disco 30 GB scsi0, máquina i440fx, rede em bridge",
    "conteineres": "Docker + Compose; chromadb/chroma:1.5.9 e qdrant/qdrant:v1.20.0, sem limite "
                   "de CPU/RAM, dados em bind mount em /root/*-data",
    "cliente": "PREENCHER (máquina com GPU NVIDIA e Ollama, na mesma rede das VMs); os bancos "
               "são testados um de cada vez",
}
# Outras VMs do mesmo host (opcional): se estiverem ligadas, disputam a CPU com o teste.
# BDA_OUTRAS_VMS="nome1=ip1,nome2=ip2". Vazio = não verifica.
OUTRAS_VMS_LAB = dict(
    item.split("=", 1) for item in os.environ.get("BDA_OUTRAS_VMS", "").split(",") if "=" in item
)


def estado_outras_vms() -> dict:
    est = {}
    for nome, ip in OUTRAS_VMS_LAB.items():
        r = subprocess.run(["ping", "-c1", "-W1", ip], capture_output=True)
        est[nome] = "ligada" if r.returncode == 0 else "desligada/sem resposta"
    return est


# ---------------------------------------------------------------- experimento
N_CORPUS = int(os.environ.get("BDA_N", "100000"))  # vetores que entram nos bancos
N_CONSULTAS = int(os.environ.get("BDA_NQ", "1000"))  # vetores separados como consulta
DIM = 1024  # bge-m3
K = 10  # top-k avaliado
N_CATS = 10  # categorias sintéticas para o teste com filtro
SEED = 42

COLECAO = "wiki_pt"
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100
EF_PADRAO = 100  # ef_search usado nas medições de latência/QPS
EF_VARREDURA = [16, 32, 64, 128, 256]  # curva recall x latência
LOTE_INGESTAO = 500
LOTE_EMBED = 64
RODADAS = int(os.environ.get("BDA_RODADAS", "3"))
AQUECIMENTO = 50
THREADS_QPS = [1, 4, 8, 16]

ARQ_CHUNKS = DADOS / "chunks.parquet"
ARQ_VETORES = DADOS / "vetores.npy"
ARQ_GABARITO = DADOS / "gabarito.npy"
ARQ_GABARITO_FILTRO = DADOS / "gabarito_filtro.npy"


def agora() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


# ---------------------------------------------------------------- log em arquivo
class _Tee:
    def __init__(self, original, arquivo):
        self.original, self.arquivo = original, arquivo

    def write(self, s):
        self.original.write(s)
        self.arquivo.write(s)
        self.arquivo.flush()

    def flush(self):
        self.original.flush()
        self.arquivo.flush()


def iniciar_log(nome: str) -> Path:
    """Duplica stdout/stderr para evidencias/logs/<timestamp>_<nome>.log."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    caminho = EVID / "logs" / f"{ts}_{nome}.log"
    f = open(caminho, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    print(f"[{agora()}] início: {nome} | comando: {' '.join(sys.argv)} | log: {caminho}")
    return caminho


def evento(script: str, fase: str, acao: str, extra: str = "") -> None:
    """Marca início/fim de fase em evidencias/eventos.csv (usado para alinhar gráficos)."""
    arq = EVID / "eventos.csv"
    novo = not arq.exists()
    with open(arq, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if novo:
            w.writerow(["timestamp", "script", "fase", "acao", "extra"])
        w.writerow([agora(), script, fase, acao, extra])
    print(f"[{agora()}] {fase}: {acao} {extra}".rstrip())


# ---------------------------------------------------------------- ssh nas VMs
def ssh_args(ip: str) -> list[str]:
    return [
        "ssh", "-i", SSH_KEY, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
        f"{SSH_USER}@{ip}",
    ]


def ssh(ip: str, comando: str, timeout: int = 60) -> str:
    r = subprocess.run(ssh_args(ip) + [comando], capture_output=True, text=True, timeout=timeout)
    return (r.stdout + r.stderr).strip()


# ---------------------------------------------------------------- monitores
class MonitorGPU:
    """nvidia-smi a cada 1 s gravando CSV em evidencias/gpu/."""

    CAMPOS = ("timestamp,name,utilization.gpu,utilization.memory,memory.used,"
              "memory.total,power.draw,temperature.gpu,clocks.sm")

    def __init__(self, rotulo: str):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.caminho = EVID / "gpu" / f"{ts}_{rotulo}.csv"

    def __enter__(self):
        self.f = open(self.caminho, "w", encoding="utf-8")
        self.p = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={self.CAMPOS}", "--format=csv", "-l", "1"],
            stdout=self.f, stderr=subprocess.STDOUT,
        )
        print(f"[{agora()}] monitor de GPU ligado -> {self.caminho}")
        return self

    def __exit__(self, *exc):
        self.p.terminate()
        self.p.wait(timeout=10)
        self.f.close()
        print(f"[{agora()}] monitor de GPU desligado")


class MonitorDocker:
    """`docker stats` contínuo dentro da VM, via uma única sessão ssh, gravando CSV."""

    def __init__(self, nome_vm: str, rotulo: str):
        self.ip, self.container, _ = VMS[nome_vm]
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.caminho = EVID / "docker" / f"{ts}_{rotulo}_{nome_vm}.csv"

    def __enter__(self):
        self.f = open(self.caminho, "w", encoding="utf-8")
        self.f.write("timestamp;container;cpu;mem_uso;mem_pct;net_io;block_io\n")
        self.f.flush()
        remoto = (
            "while true; do docker stats --no-stream --format "
            "'{{.Name}};{{.CPUPerc}};{{.MemUsage}};{{.MemPerc}};{{.NetIO}};{{.BlockIO}}' "
            f"{self.container} | sed \"s/^/$(date -Iseconds);/\"; done"
        )
        self.p = subprocess.Popen(ssh_args(self.ip) + [remoto], stdout=self.f,
                                  stderr=subprocess.DEVNULL)
        print(f"[{agora()}] monitor docker stats ligado ({self.container}) -> {self.caminho}")
        return self

    def __exit__(self, *exc):
        self.p.terminate()
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.f.close()
        print(f"[{agora()}] monitor docker stats desligado ({self.container})")
