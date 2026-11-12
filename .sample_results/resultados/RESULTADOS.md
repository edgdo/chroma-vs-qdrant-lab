# Resultados: Chroma x Qdrant

Gerado em 2026-11-10T21:09:24.058-03:00. Execução do benchmark: única. Valores = mediana das rodadas.

## Ambiente do teste

| Item | Valor |
|---|---|
| hipervisor | Proxmox VE, nó <no-proxmox> (versão: PREENCHER com pveversion) |
| host_cpu | Intel Core i7-7700K @ 4.20 GHz (4 núcleos / 8 threads, L3 8 MB) |
| host_memoria | PREENCHER (free -h no host) |
| host_storage | datastore vmdata: PREENCHER (tipo e disco: pvesm status; SSD/NVMe/HDD, ZFS/LVM) |
| provisionamento | Terraform (provider bpg/proxmox 0.120.0), módulo ./module, clone do template 9000 (Ubuntu 24.04 cloud-init) |
| vms | 2 VMs idênticas (bda-chroma, bda-qdrant): 2 vCPU tipo host, 4 GiB RAM dedicada, disco 30 GB scsi0 no datastore vmdata, máquina i440fx, rede em bridge |
| conteineres | Docker + Compose; chromadb/chroma:1.5.9 e qdrant/qdrant:v1.20.0, sem limite de CPU/RAM, dados em bind mount em /root/*-data |
| cliente | WSL2 (Ubuntu) no PC com RTX 4080, mesma LAN <lan>/24; os bancos são testados um de cada vez |
| GPU do cliente | NVIDIA GeForce RTX 4080, 16376 MiB, 610.88 |
| CPU do cliente | AMD Ryzen 7 9800X3D 8-Core Processor |
| VM chroma | Ubuntu 24.04.5 LTS, kernel 6.8.0-147-generic, 2 vCPU Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz, mem Mem:           3.8Gi       581Mi       1.1Gi       1.1Mi       2.5Gi       3.3Gi, imagem `chromadb/chroma:1.5.9` |
| VM qdrant | Ubuntu 24.04.5 LTS, kernel 6.8.0-147-generic, 2 vCPU Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz, mem Mem:           3.8Gi       576Mi       1.6Gi       1.1Mi       1.9Gi       3.3Gi, imagem `qdrant/qdrant:v1.20.0` |
| Outras VMs no host (manifesto) | {'vm-vizinha-1': 'ligada', 'vm-vizinha-2': 'ligada', 'vm-vizinha-3': 'ligada', 'vm-vizinha-4': 'ligada'} |
| Pacotes Python | {'chromadb-client': '1.5.9', 'qdrant-client': '1.20.0', 'numpy': '2.5.4', 'pandas': '3.1.1', 'datasets': '5.2.0', 'matplotlib': '3.11.2', 'requests': '2.34.2'} |

## Ingestão

| banco | insert_s | indice_pronto_s | vetores_por_s |
|---|---|---|---|
| chroma | 154.3 | 154.3 | 648.1 |
| qdrant | 85.1 | 131.6 | 759.8 |

## Latência e recall (ef padrão)

| banco | cenario | p50_ms | p95_ms | p99_ms | recall |
|---|---|---|---|---|---|
| chroma | com_filtro | 49.278 | 54.907 | 68.059 | 0.999 |
| chroma | sem_filtro | 2.372 | 2.646 | 2.898 | 0.991 |
| qdrant | com_filtro | 2.258 | 2.475 | 2.634 | 0.999 |
| qdrant | sem_filtro | 2.435 | 2.715 | 3.024 | 0.996 |

## Varredura de ef_search

| banco | ef | p95_ms | recall |
|---|---|---|---|
| chroma | 16 | 2.296 | 0.9189 |
| chroma | 32 | 2.358 | 0.9644 |
| chroma | 64 | 2.599 | 0.9868 |
| chroma | 128 | 2.972 | 0.9945 |
| chroma | 256 | 3.696 | 0.9981 |
| qdrant | 16 | 2.086 | 0.9343 |
| qdrant | 32 | 2.241 | 0.9764 |
| qdrant | 64 | 2.535 | 0.9923 |
| qdrant | 128 | 3.046 | 0.9969 |
| qdrant | 256 | 3.982 | 0.9988 |

## Throughput (QPS)

| threads | chroma | qdrant |
|---|---|---|
| 1 | 389.1 | 393.9 |
| 4 | 683.0 | 789.2 |
| 8 | 710.1 | 939.8 |
| 16 | 706.7 | 962.6 |

## Disco

| banco | mb |
|---|---|
| chroma | 1673.6 |
| qdrant | 670.6 |

## Recursos (pico durante o benchmark)

| banco | cpu_pico_% | ram_pico_MB |
|---|---|---|
| chroma | 130.25 | 2316.0 |
| qdrant | 201.32 | 778.0 |

## GPU (evidência de uso)

- GPU `20261110-183749_embeddings`: 1161 s, uso médio 40%, pico 64%, VRAM pico 6808 MiB, potência pico 156 W
- GPU `20261110-190151_embeddings`: 1 s, uso médio 5%, pico 5%, VRAM pico 6083 MiB, potência pico 60 W
