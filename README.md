# chroma-vs-qdrant-lab: Chroma x Qdrant com embeddings na GPU

Benchmark de bancos vetoriais feito para um trabalho acadêmico (MBA FIAP, disciplina de Big Data
Architecture). Mesmos dados, mesmos vetores, mesmo HNSW (cosseno, M=16, ef_construct=100,
ef_search=100) e mesmo cliente; só o banco muda.

| Peça | Onde |
|---|---|
| Chroma 1.5.9 | servidor próprio (`BDA_CHROMA_HOST`, porta 8000) |
| Qdrant 1.20.0 | servidor próprio (`BDA_QDRANT_URL`, porta 6333) |
| Embeddings `bge-m3` (1024 dims) e LLM `qwen2.5:14b` | Ollama com GPU NVIDIA na máquina cliente |
| Scripts, dados e evidências | esta pasta |

## Como o ambiente foi montado

Os bancos **não rodaram na máquina local**. Cada um ficou numa VM própria, num hipervisor
Proxmox VE da rede local, e o cliente (scripts + Ollama na GPU) ficou num PC separado na mesma
rede. Assim a GPU e os scripts não disputam CPU e memória com o banco medido, e os dois bancos
têm exatamente o mesmo hardware.

### As VMs

Duas VMs idênticas, criadas com Terraform (provider `bpg/proxmox`) a partir do mesmo template:

| Item | Valor |
|---|---|
| Nomes | `bda-chroma` e `bda-qdrant` |
| Sistema | Ubuntu 24.04, clone completo de um template cloud-init |
| vCPU | 2, tipo `host` (repassa as instruções da CPU física, inclusive AVX) |
| Memória | 4 GiB dedicada (sem balloon) |
| Disco | 30 GB, `scsi0`, mesmo datastore para as duas |
| Rede | bridge na LAN, IP fixo configurado pelo cloud-init |
| Acesso | usuário criado pelo cloud-init com chave SSH pública (sem senha) |
| Docker | instalado no primeiro boot por um snippet de cloud-init (vendor data) |

O trecho do Terraform que cria cada VM, em resumo:

```hcl
resource "proxmox_virtual_environment_vm" "bda_chroma" {   # igual para bda_qdrant
  name      = "bda-chroma"
  node_name = "<seu-no>"
  clone { vm_id = <id-do-template>  full = true }
  cpu    { cores = 2  type = "host" }
  memory { dedicated = 4096 }
  disk   { datastore_id = "<datastore>"  interface = "scsi0"  size = 30 }
  network_device { bridge = "vmbr0" }
  agent  { enabled = true }
  initialization {
    ip_config { ipv4 { address = "<ip>/24"  gateway = "<gateway>" } }
    user_account { username = "<usuario>"  keys = [<chave-publica>] }
    vendor_data_file_id = "local:snippets/<snippet-que-instala-docker>.yaml"
  }
}
```

Sem Proxmox, vale qualquer forma de ter duas máquinas iguais (outro hipervisor, nuvem, físicas).
O importante é que sejam idênticas e que só um banco esteja sob carga por vez. Se houver outras
VMs no mesmo host, desligue-as durante o `04_benchmark.py` (ou registre-as em `BDA_OUTRAS_VMS`).

### Instalar os bancos

Em cada VM, um `docker-compose.yml` em `/root`, sem limite de CPU/RAM e com os dados em bind mount
numa pasta local (é ela que o benchmark mede com `du`). As imagens usadas foram
`chromadb/chroma:1.5.9` e `qdrant/qdrant:v1.20.0`.

Docker, se o cloud-init não instalou:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER   # o usuário SSH precisa rodar docker sem sudo
```

VM do Chroma (`/root/docker-compose.yml`):

```yaml
services:
  chroma:
    image: chromadb/chroma:1.5.9
    container_name: chroma
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - /root/chroma-data:/data
```

VM do Qdrant (`/root/docker-compose.yml`):

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.20.0
    container_name: qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - /root/qdrant-data:/qdrant/storage
```

Subir e conferir:

```bash
sudo -i
cd /root && docker compose up -d
curl -s localhost:8000/api/v2/version   # Chroma
curl -s localhost:6333                  # Qdrant
```

Os nomes `chroma` e `qdrant` dos containers e as pastas `/root/*-data` são os que os scripts
esperam (`VMS` em `comum.py`). O usuário SSH também precisa de `sudo du` sem senha, para medir o
disco.

Telas úteis depois da carga: dashboard do Qdrant em `http://<ip-do-qdrant>:6333/dashboard` e
Swagger do Chroma em `http://<ip-do-chroma>:8000/docs/`.

### Máquina cliente (origem dos testes)

Os scripts, o Ollama e a GPU ficaram num PC Windows com WSL2, na mesma rede das VMs. Valores
tirados do manifesto da rodada (`00_manifesto.py`):

| Item | Valor usado |
|---|---|
| CPU | AMD Ryzen 7 9800X3D (8 núcleos) |
| GPU (VGA) | NVIDIA GeForce RTX 4080, 16 GB de VRAM |
| Driver NVIDIA | 610.88 (instalado no Windows; o WSL2 usa o mesmo driver, sem instalar nada no Linux) |
| Sistema | Windows 11 + WSL2 com Ubuntu 24.04 (kernel 6.6.87.2-microsoft-standard-WSL2) |
| Python | 3.12.3, em venv |
| Ollama | 0.23.1, instalado dentro do WSL, rodando 100% na GPU |
| Modelo de embedding | `bge-m3` (567M parâmetros, F16, 1024 dimensões), ~1,2 GB de VRAM |
| LLM do RAG | `qwen2.5:14b` (14,8B parâmetros, Q4_K_M) |
| Pacotes Python | chromadb-client 1.5.9, qdrant-client 1.20.0, numpy 2.5.4, pandas 3.1.1, datasets 5.2.0, matplotlib 3.11.2, requests 2.34.2 |

Requisitos mínimos para reproduzir: GPU NVIDIA com pelo menos ~12 GB de VRAM para o
`qwen2.5:14b` (o `bge-m3` sozinho cabe em bem menos), `nvidia-smi` funcionando (os scripts gravam
uso da GPU a cada segundo), Python 3.12, Ollama, cliente `ssh` e `ping`. Funciona igual num Linux
nativo; o WSL2 só foi o ambiente usado aqui.

Preparar o cliente no WSL2 (Ubuntu):

```bash
# GPU: instale o driver NVIDIA no Windows; no WSL confira com
nvidia-smi

# Python e ferramentas
sudo apt update && sudo apt install -y python3 python3-venv python3-pip openssh-client iputils-ping

# Ollama e modelos
curl -fsSL https://ollama.com/install.sh | sh
ollama pull bge-m3
ollama pull qwen2.5:14b
ollama ps   # durante o uso, a coluna PROCESSOR deve mostrar 100% GPU
```

Uma dica do WSL: deixe o venv e a chave SSH no disco do Linux (`~`), não numa pasta do Windows
montada em `/mnt/c`. Lá os arquivos aparecem com permissão aberta, o `ssh` recusa a chave
("UNPROTECTED PRIVATE KEY FILE") e binários do venv podem não executar.

## Configuração

Tudo é ajustável por variável de ambiente, com padrões em `comum.py`:

| Variável | Padrão | Para quê |
|---|---|---|
| `BDA_CHROMA_HOST` / `BDA_CHROMA_PORT` | `localhost` / `8000` | Servidor do Chroma |
| `BDA_QDRANT_URL` | `http://localhost:6333` | Servidor do Qdrant |
| `BDA_SSH_USER` | `labuser` | Usuário SSH nos servidores dos bancos |
| `BDA_SSH_KEY` | `~/.ssh/lab` | Chave privada usada pelos scripts (`-i`) |
| `BDA_OLLAMA_URL` | `http://localhost:11434` | Ollama |
| `BDA_EMBED_MODEL` / `BDA_LLM_MODEL` | `bge-m3` / `qwen2.5:14b` | Modelos |
| `BDA_DADOS` | `./dados` | Pasta dos dados pesados (~1 GB) |
| `BDA_OUTRAS_VMS` | vazio | Opcional: `nome=ip,nome=ip` de VMs vizinhas no mesmo host, para registrar se estavam ligadas |
| `BDA_N`, `BDA_NQ`, `BDA_RODADAS` | `100000`, `1000`, `3` | Tamanho do corpus, nº de consultas e rodadas |

Antes da rodada valendo, preencha o dicionário `INFRA` em `comum.py` com a descrição do seu
ambiente (hipervisor, CPU, memória, storage, cliente). Ele vai para o manifesto e para o
`RESULTADOS.md`.

## Preparar o ambiente (uma vez)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export BDA_CHROMA_HOST=<ip-do-chroma>
export BDA_QDRANT_URL=http://<ip-do-qdrant>:6333
export BDA_SSH_USER=<usuario>
export BDA_SSH_KEY=~/.ssh/<sua-chave>
```

## Pré-requisito: SSH nos servidores dos bancos

O manifesto e o benchmark entram por SSH nos servidores para coletar versões, disco e
`docker stats`, e para reiniciar o container do Chroma quando o `ef_search` muda. A chave precisa
funcionar sem senha:

```bash
ssh -o BatchMode=yes -i "$BDA_SSH_KEY" "$BDA_SSH_USER@$BDA_CHROMA_HOST" hostname
```

Os scripts rodam com `BatchMode=yes` e `StrictHostKeyChecking=accept-new`. Se um servidor for
recriado no mesmo IP, a host key muda e o SSH bloqueia: rode `ssh-keygen -R <IP>`.

Nos servidores, os containers se chamam `chroma` e `qdrant`, com dados em `/root/chroma-data` e
`/root/qdrant-data` (ajuste `VMS` em `comum.py` se for diferente). O usuário SSH precisa de
`docker` e de `sudo du` sem senha.

## Rodar, na ordem

| # | Comando | O que faz | Tempo aprox. |
|---|---|---|---|
| 0 | `python 00_manifesto.py` | Registra versões (host, GPU, Ollama, libs, servidores, imagens) | 10 s |
| 1 | `python 01_preparar_dados.py` | Baixa Wikipédia-pt (streaming) e gera 101 mil chunks | 2-5 min |
| 2 | `python 02_gerar_embeddings.py` | bge-m3 na GPU, com nvidia-smi gravando | 10-20 min |
| 3 | `python 03_gabarito.py` | Top-10 exato (força bruta) = gabarito do recall | < 1 min |
| 4 | `python 04_benchmark.py` | 3 rodadas x 2 bancos: ingestão, latência, recall, filtro, QPS, disco | 30-60 min |
| 5 | `python 05_rag_demo.py` | RAG com a mesma pergunta nos dois bancos (GPU de novo) | 1-2 min |
| 6 | `python 06_graficos.py` | Gráficos + `resultados/RESULTADOS.md` | 10 s |
| 0 | `python 00_manifesto.py` | Manifesto final (com hash dos dados e disco dos servidores) | 10 s |

Para um teste rápido antes do completo: `BDA_N=5000 BDA_NQ=200 BDA_RODADAS=1`, prefixando os
passos 1 a 4. Depois apague `dados/`, `resultados/` e `evidencias/` antes da rodada valendo.

Perguntas próprias no RAG: `python 05_rag_demo.py "sua pergunta" "outra pergunta"`.

## O que fica de evidência

- `evidencias/logs/`: saída completa de cada script, com data/hora e comando
- `evidencias/gpu/`: nvidia-smi a cada 1 s (uso %, VRAM, potência, temperatura, clock)
- `evidencias/docker/`: `docker stats` de cada container durante o benchmark
- `evidencias/ollama_ps.txt`: modelo carregado e % na GPU
- `evidencias/eventos.csv`: início/fim de cada fase (para alinhar com os gráficos)
- `evidencias/manifesto_*.json`: versões e hashes
- `evidencias/rag_demo_*.md`: perguntas, trechos recuperados e respostas
- `resultados/*.csv` + `*.png` + `RESULTADOS.md`

As pastas `dados/`, `evidencias/` e `resultados/` são geradas na execução e ficam fora do git.

## Resultados de exemplo

`.sample_results/` traz a rodada real feita com o ambiente descrito acima: `resultados/` (CSVs,
latências brutas em `.npy`, gráficos e `RESULTADOS.md`) e `evidencias/` (logs, manifestos,
`nvidia-smi`, `docker stats`, eventos, demo de RAG e prints). Caminhos locais, IPs e nomes de
outras VMs foram trocados por placeholders (`<repo>`, `<ip-chroma>`, `<ip-qdrant>`, `<lan>`,
`vm-vizinha-N`) e os prints tiveram esses trechos tarjados.

## Zerar

O `04_benchmark.py` já recria as coleções a cada rodada. Para limpar tudo numa VM:
`sudo -i`, depois `cd /root && docker compose down && rm -rf /root/*-data && docker compose up -d`.
