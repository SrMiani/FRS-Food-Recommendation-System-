"""
Evaluación del retrieval de PlatePicker contra el golden set.

Uso (desde FRS/gestion/RAG):
    python eval/evaluate_retrieval.py

Necesita:
    - eval/golden_set.json
    - el índice ya construido con multimodal_index.py (chroma_multimodal en la raíz de FRS)
"""

import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor


# =====================================================
# CONFIG
# =====================================================
EVAL_DIR     = Path(__file__).resolve().parent          # .../FRS/gestion/RAG/eval
PROJECT_ROOT = EVAL_DIR.parents[2]                      # .../FRS
DB_DIR       = str(PROJECT_ROOT / "chroma_multimodal")
GOLDEN_PATH  = EVAL_DIR / "golden_set.json"
RESULTS_DIR  = EVAL_DIR / "results"

ARTICLE_COLLECTION = "restaurant_articles"
IMAGE_COLLECTION   = "recipe_images"

K_VALUES = [1, 3, 5, 10]     # los top-k que vamos a comparar
MAX_K    = max(K_VALUES)

TEXT_MODEL_NAME = "all-MiniLM-L6-v2"   # <- aquí cambiarás el modelo en los experimentos
RUN_NAME        = "baseline_minilm"    # <- nombre del experimento (sale en el CSV)


# =====================================================
# MÉTRICAS  ->  ESTAS DOS LAS ESCRIBES TÚ
# =====================================================
def hit_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> int:
    """
    Devuelve 1 si ALGUNO de los expected_ids aparece entre los k primeros retrieved_ids.
    Si no, devuelve 0.

    Ejemplo:
        retrieved = ["restaurant_7", "restaurant_3", "restaurant_9"]
        expected  = ["restaurant_3"]
        hit_at_k(retrieved, expected, 1) -> 0   (el primero es el 7)
        hit_at_k(retrieved, expected, 3) -> 1   (el 3 está entre los 3 primeros)
    """
    # TODO: escribe aquí tu código
    raise NotImplementedError("Escribe hit_at_k")


def reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """
    Busca la PRIMERA posición (empezando en 1) en la que aparece alguno de los expected_ids.
    Devuelve 1 / posición. Si no aparece ninguno, devuelve 0.0

    Ejemplo:
        retrieved = ["restaurant_7", "restaurant_3", "restaurant_9"]
        expected  = ["restaurant_3"]
        reciprocal_rank(retrieved, expected) -> 0.5   (está en la posición 2 -> 1/2)
    """
    # TODO: escribe aquí tu código
    raise NotImplementedError("Escribe reciprocal_rank")


# =====================================================
# MODELOS DE EMBEDDINGS
# =====================================================
print(f"Cargando modelos ({TEXT_MODEL_NAME} + CLIP)...")
text_model = SentenceTransformer(TEXT_MODEL_NAME)

device = "cpu"
clip_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
clip_model.eval()


def embed_text_query(query: str) -> list[float]:
    vec = text_model.encode([query], normalize_embeddings=True)[0]
    return vec.astype(np.float32).tolist()


@torch.no_grad()
def embed_clip_text_query(query: str) -> list[float]:
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True, truncation=True).to(device)
    feats = clip_model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats[0].cpu().numpy().astype(np.float32).tolist()


# =====================================================
# RETRIEVAL
# =====================================================
article_db = Chroma(collection_name=ARTICLE_COLLECTION, persist_directory=DB_DIR)
image_db   = Chroma(collection_name=IMAGE_COLLECTION,   persist_directory=DB_DIR)

print(f"Vectores: {article_db._collection.count()} restaurantes, {image_db._collection.count()} imágenes")


def search(item: dict):
    """Lanza una consulta del golden set y devuelve (ids, similitudes, segundos)."""
    start = time.perf_counter()
    if item["collection"] == "images":
        q_vec = embed_clip_text_query(item["query"])
        db = image_db
    else:
        q_vec = embed_text_query(item["query"])
        db = article_db

    res = db._collection.query(
        query_embeddings=[q_vec],
        n_results=MAX_K,
        where=item.get("where"),
        include=["distances"],
    )
    elapsed = time.perf_counter() - start

    ids = res["ids"][0]
    # Chroma usa distancia L2 al cuadrado; con vectores normalizados: coseno = 1 - d/2
    sims = [1 - d / 2 for d in res["distances"][0]]
    return ids, sims, elapsed


# =====================================================
# EVALUACIÓN
# =====================================================
golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
print(f"Golden set: {len(golden)} consultas\n")

rows = []
for item in golden:
    ids, sims, secs = search(item)
    row = {
        "id": item["id"],
        "type": item["type"],
        "query": item["query"],
        "expected": "|".join(item["expected"]),
        "top1": ids[0] if ids else "",
        "top1_sim": round(sims[0], 4) if sims else 0.0,
        "latency_ms": round(secs * 1000, 1),
    }
    if item["expected"]:  # consultas con respuesta
        for k in K_VALUES:
            row[f"hit@{k}"] = hit_at_k(ids, item["expected"], k)
        row["rr"] = round(reciprocal_rank(ids, item["expected"]), 4)
    rows.append(row)


# =====================================================
# RESUMEN
# =====================================================
def mean(values):
    return sum(values) / len(values) if values else 0.0

answerable = [r for r in rows if "rr" in r]
no_answer  = [r for r in rows if "rr" not in r]

by_type = defaultdict(list)
for r in answerable:
    by_type[r["type"]].append(r)

header = f"{'tipo':<16}{'n':>3}" + "".join(f"{'hit@'+str(k):>8}" for k in K_VALUES) + f"{'MRR':>8}"
print(header)
print("-" * len(header))
for t, rs in sorted(by_type.items()):
    line = f"{t:<16}{len(rs):>3}" + "".join(f"{mean([r[f'hit@{k}'] for r in rs]):>8.2f}" for k in K_VALUES)
    print(line + f"{mean([r['rr'] for r in rs]):>8.2f}")
print("-" * len(header))
line = f"{'TOTAL':<16}{len(answerable):>3}" + "".join(f"{mean([r[f'hit@{k}'] for r in answerable]):>8.2f}" for k in K_VALUES)
print(line + f"{mean([r['rr'] for r in answerable]):>8.2f}")

print(f"\nLatencia media: {mean([r['latency_ms'] for r in rows]):.1f} ms por consulta")

# Para decidir el umbral de "no hay match": compara la similitud del top-1
print("\nSimilitud del top-1 (para elegir el umbral de 'no hay match'):")
ok_sims = [r["top1_sim"] for r in answerable if r.get("hit@1") == 1]
print(f"  consultas con acierto en top-1 -> min {min(ok_sims, default=0):.3f} | media {mean(ok_sims):.3f}")
for r in no_answer:
    print(f"  trampa {r['id']} ({r['query'][:40]}) -> {r['top1_sim']:.3f}")

print("\nFallos (el correcto no está en el top-5):")
for r in answerable:
    if r["hit@5"] == 0:
        print(f"  {r['id']} [{r['type']}] {r['query'][:60]}  -> top1: {r['top1']}  (esperado: {r['expected']})")


# =====================================================
# GUARDAR CSV
# =====================================================
RESULTS_DIR.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M")
out = RESULTS_DIR / f"{stamp}_{RUN_NAME}.csv"
fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0].keys()).index(k) if k in rows[0] else 99)
with open(out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"\nResultados guardados en {out}")
