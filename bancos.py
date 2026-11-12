"""Adaptadores: a MESMA interface para Chroma e Qdrant.

O código de medição (04_benchmark.py) só fala com esta interface, então a única coisa que
muda entre os dois testes é o banco. Os dois guardam a mesma coisa por ponto: vetor, texto,
título e categoria; e usam HNSW com os mesmos parâmetros (cosseno, M, ef_construct, ef_search).
"""
from __future__ import annotations

import time

import chromadb
import numpy as np
import requests
from qdrant_client import QdrantClient, models

from comum import (CHROMA_HOST, CHROMA_PORT, COLECAO, DIM, EF_PADRAO, HNSW_EF_CONSTRUCT, HNSW_M,
                   QDRANT_URL, VMS, ssh)


class Chroma:
    nome = "chroma"

    def __init__(self) -> None:
        self.c = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        self.col = None

    def versao(self) -> str:
        return requests.get(f"http://{CHROMA_HOST}:{CHROMA_PORT}/api/v2/version", timeout=10).text

    def recriar(self) -> None:
        try:
            self.c.delete_collection(COLECAO)
        except Exception:  # noqa: BLE001  (não existia)
            pass
        self.col = self.c.create_collection(
            COLECAO,
            configuration={"hnsw": {"space": "cosine", "max_neighbors": HNSW_M,
                                    "ef_construction": HNSW_EF_CONSTRUCT,
                                    "ef_search": EF_PADRAO}},
        )

    def abrir(self) -> None:
        self.col = self.c.get_collection(COLECAO)

    def inserir(self, ids, vetores: np.ndarray, metas: list[dict], textos: list[str]) -> None:
        self.col.add(ids=[str(i) for i in ids], embeddings=vetores, metadatas=metas,
                     documents=textos)

    def aguardar_indice(self) -> dict:
        # o Chroma indexa durante o add; não há fila de indexação assíncrona para esperar
        return {"pontos": self.col.count()}

    def set_ef(self, ef: int) -> None:
        """Muda o ef_search do Chroma.

        No Chroma 1.5.9 o modify() grava a configuração nova, mas o índice HNSW já carregado em
        memória continua usando o ef da criação (testado: recall igual antes e depois). O valor
        só passa a valer quando o servidor recarrega o índice, então reiniciamos o container.
        """
        atual = (self.col.configuration_json or {}).get("hnsw", {}).get("ef_search")
        if atual == ef:
            return
        self.col.modify(configuration={"hnsw": {"ef_search": ef}})
        ip, container, _ = VMS["chroma"]
        ssh(ip, f"docker restart {container} >/dev/null", timeout=120)
        for _ in range(120):
            try:
                self.c = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
                self.c.heartbeat()
                break
            except Exception:  # noqa: BLE001  (subindo)
                time.sleep(1)
        self.col = self.c.get_collection(COLECAO)

    def buscar(self, vetor: np.ndarray, k: int, cat: str | None = None) -> list[int]:
        r = self.col.query(query_embeddings=[vetor], n_results=k,
                           where={"cat": cat} if cat else None, include=["distances"])
        return [int(x) for x in r["ids"][0]]

    def buscar_com_texto(self, vetor: np.ndarray, k: int) -> list[dict]:
        r = self.col.query(query_embeddings=[vetor], n_results=k,
                           include=["documents", "metadatas", "distances"])
        return [{"id": int(i), "titulo": m["titulo"], "texto": d, "score": 1 - dist}
                for i, m, d, dist in zip(r["ids"][0], r["metadatas"][0], r["documents"][0],
                                         r["distances"][0])]

    def contar(self) -> int:
        return self.col.count()


class Qdrant:
    nome = "qdrant"

    def __init__(self) -> None:
        self.c = QdrantClient(url=QDRANT_URL, timeout=300)
        self.ef = EF_PADRAO

    def versao(self) -> str:
        return requests.get(QDRANT_URL, timeout=10).json()["version"]

    def recriar(self) -> None:
        if self.c.collection_exists(COLECAO):
            self.c.delete_collection(COLECAO)
        self.c.create_collection(
            COLECAO,
            vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
        )
        # índice de payload para o filtro por categoria (equivalente ao índice de metadado)
        self.c.create_payload_index(COLECAO, field_name="cat",
                                    field_schema=models.PayloadSchemaType.KEYWORD)
        self.ef = EF_PADRAO

    def abrir(self) -> None:
        pass

    def inserir(self, ids, vetores: np.ndarray, metas: list[dict], textos: list[str]) -> None:
        payloads = [{**m, "texto": t} for m, t in zip(metas, textos)]
        self.c.upsert(COLECAO, points=models.Batch(ids=[int(i) for i in ids],
                                                   vectors=vetores.tolist(), payloads=payloads),
                      wait=True)

    def aguardar_indice(self) -> dict:
        """O Qdrant constrói o HNSW em segundo plano; espera a coleção ficar verde."""
        time.sleep(2)
        while True:
            info = self.c.get_collection(COLECAO)
            if info.status == models.CollectionStatus.GREEN:
                return {"pontos": info.points_count,
                        "vetores_indexados": info.indexed_vectors_count,
                        "segmentos": info.segments_count}
            time.sleep(1)

    def set_ef(self, ef: int) -> None:
        self.ef = ef

    def _filtro(self, cat: str | None):
        if not cat:
            return None
        return models.Filter(must=[models.FieldCondition(key="cat",
                                                         match=models.MatchValue(value=cat))])

    def buscar(self, vetor: np.ndarray, k: int, cat: str | None = None) -> list[int]:
        r = self.c.query_points(COLECAO, query=vetor.tolist(), limit=k,
                                search_params=models.SearchParams(hnsw_ef=self.ef),
                                query_filter=self._filtro(cat), with_payload=False)
        return [int(p.id) for p in r.points]

    def buscar_com_texto(self, vetor: np.ndarray, k: int) -> list[dict]:
        r = self.c.query_points(COLECAO, query=vetor.tolist(), limit=k,
                                search_params=models.SearchParams(hnsw_ef=self.ef),
                                with_payload=True)
        return [{"id": int(p.id), "titulo": p.payload["titulo"], "texto": p.payload["texto"],
                 "score": p.score} for p in r.points]

    def contar(self) -> int:
        return self.c.count(COLECAO, exact=True).count


BANCOS = {"chroma": Chroma, "qdrant": Qdrant}
