"""Passo 0 (rode no início e no fim): registra versões e ambiente em evidencias/manifesto_<ts>.json.

Não lê nem grava segredo nenhum: só versões, hardware e hash dos arquivos de dados.
"""
import hashlib
import json
import platform
import subprocess
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

import requests

from comum import (CHROMA_HOST, CHROMA_PORT, DADOS, EMBED_MODEL, EVID, INFRA, LLM_MODEL,
                   OLLAMA_URL, QDRANT_URL, VMS, agora, estado_outras_vms, iniciar_log, ssh)


def cmd(*a: str) -> str:
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"erro: {e}"


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def main() -> None:
    iniciar_log("00_manifesto")
    m: dict = {"gerado_em": agora(), "infra": INFRA, "outras_vms_no_host": estado_outras_vms()}

    m["host"] = {
        "plataforma": platform.platform(),
        "python": platform.python_version(),
        "cpu": cmd("bash", "-c", "lscpu | grep 'Model name' | cut -d: -f2").strip(),
        "gpu": cmd("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                   "--format=csv,noheader"),
    }
    pacotes = ["chromadb-client", "qdrant-client", "numpy", "pandas", "datasets", "matplotlib",
               "requests"]
    m["pacotes_python"] = {}
    for p in pacotes:
        try:
            m["pacotes_python"][p] = version(p)
        except PackageNotFoundError:
            m["pacotes_python"][p] = None

    try:
        m["ollama"] = {
            "versao": requests.get(f"{OLLAMA_URL}/api/version", timeout=10).json(),
            "modelos": {
                n: requests.post(f"{OLLAMA_URL}/api/show", json={"model": n}, timeout=30)
                .json().get("details") for n in (EMBED_MODEL, LLM_MODEL)
            },
        }
    except Exception as e:  # noqa: BLE001
        m["ollama"] = f"erro: {e}"

    m["servidores"] = {}
    try:
        m["servidores"]["chroma"] = {
            "versao": requests.get(f"http://{CHROMA_HOST}:{CHROMA_PORT}/api/v2/version",
                                   timeout=10).json(),
        }
    except Exception as e:  # noqa: BLE001
        m["servidores"]["chroma"] = f"erro: {e}"
    try:
        m["servidores"]["qdrant"] = requests.get(QDRANT_URL, timeout=10).json()
    except Exception as e:  # noqa: BLE001
        m["servidores"]["qdrant"] = f"erro: {e}"

    m["vms"] = {}
    for nome, (ip, container, pasta) in VMS.items():
        m["vms"][nome] = {
            "ip": ip,
            "hostname": ssh(ip, "hostname"),
            "so": ssh(ip, ". /etc/os-release && echo $PRETTY_NAME"),
            "kernel": ssh(ip, "uname -r"),
            "vcpu": ssh(ip, "nproc"),
            "cpu_modelo": ssh(ip, "lscpu | grep 'Model name' | cut -d: -f2 | xargs"),
            "cpu_topologia": ssh(ip, "lscpu | grep -E 'Socket|Core|Thread|Hypervisor' | tr -s ' '"),
            "virtualizacao": ssh(ip, "systemd-detect-virt; cat /sys/class/dmi/id/product_name"),
            "memoria": ssh(ip, "free -h | sed -n 2p"),
            "disco": ssh(ip, "lsblk -d -o NAME,SIZE,ROTA,MODEL | grep -v sr0"),
            "uptime_load": ssh(ip, "uptime"),
            "docker": ssh(ip, "docker --version"),
            "imagem": ssh(ip, f"docker inspect --format '{{{{.Config.Image}}}} {{{{.Image}}}}' {container}"),
            "container": ssh(ip, f"docker ps --filter name={container} --format '{{{{.Status}}}}'"),
            "disco_dados": ssh(ip, f"sudo du -sh {pasta}"),
        }

    m["arquivos_dados"] = {
        p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(DADOS.glob("*")) if p.is_file()
    }

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    saida = EVID / f"manifesto_{ts}.json"
    saida.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(m, indent=2, ensure_ascii=False))
    print(f"\nManifesto salvo em {saida}")


if __name__ == "__main__":
    main()
