# Standard library
import glob
import json
import os
import re
import shutil
from pathlib import Path

# Third-party library
import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor


# =====================================================
# CONFIG
# =====================================================
SHOW_SCHEMA = False  # True: enseña un restaurante y una receta del JSON y para. Ponlo a False para indexar.

BASE_DIR     = Path(__file__).resolve().parent        # .../FRS/gestion/RAG
PROJECT_ROOT = BASE_DIR.parents[1]                    # .../FRS
IMG_DIR      = BASE_DIR.parent / "synthetic_recipe_images"
DB_DIR       = str(PROJECT_ROOT / "chroma_multimodal")

ARTICLE_COLLECTION = "restaurant_articles"
IMAGE_COLLECTION   = "recipe_images"   # el mismo nombre que en retriever.py y explore_data.py


def find_file(filename: str) -> Path:
    """Busca el JSON en las carpetas típicas del proyecto, da igual desde dónde ejecutes."""
    candidates = [BASE_DIR, BASE_DIR.parent, PROJECT_ROOT]
    for folder in candidates:
        path = folder / filename
        if path.exists():
            return path
    # último recurso: buscarlo en todo el proyecto
    found = list(PROJECT_ROOT.rglob(filename))
    if found:
        return found[0]
    raise FileNotFoundError(f"No encuentro {filename} dentro de {PROJECT_ROOT}")


# =====================================================
# LOAD DATA
# =====================================================
images_paths = sorted(glob.glob(str(IMG_DIR / "**" / "*.png"), recursive=True))
if not images_paths:
    raise FileNotFoundError(f"No se encontraron imágenes en {IMG_DIR}")
print(f"Se encontraron {len(images_paths)} imágenes en {IMG_DIR}.")

restaurants_path = find_file("structured_restaurant_data.json")
recipes_path     = find_file("Recipes_with_image_descriptions.json")

with open(restaurants_path, "r", encoding="utf-8") as f:
    restaurants = json.load(f)
with open(recipes_path, "r", encoding="utf-8") as f:
    recipes = json.load(f)

print(f"Se encontraron {len(restaurants)} restaurantes y {len(recipes)} recetas.")
print(f"  restaurantes: {restaurants_path}")
print(f"  recetas:      {recipes_path}")

if SHOW_SCHEMA:
    print("\n--- RESTAURANTE [0] ---")
    print(json.dumps(restaurants[0], indent=2, ensure_ascii=False)[:2000])
    print("\n--- RECETA [0] ---")
    print(json.dumps(recipes[0], indent=2, ensure_ascii=False)[:2000])
    raise SystemExit("\nSHOW_SCHEMA=True -> paro aquí. Pásale esta salida a Claude.")


# =====================================================
# HELPERS
# =====================================================
def clean(value) -> str:
    """Chroma no acepta None en metadatos: lo convertimos a texto vacío."""
    return "" if value is None else str(value).strip()


def record_to_text(record: dict, skip: set) -> str:
    """Convierte todos los campos de texto de un registro en 'campo: valor', uno por línea.
    Así el documento lleva descripción, platos, reseñas... y no solo nombre/cocina/ubicación."""
    lines = []
    for key, value in record.items():
        if key in skip or value in (None, "", [], {}):
            continue
        if isinstance(value, (str, int, float)):
            lines.append(f"{key}: {value}")
        elif isinstance(value, list):
            items = [str(v) for v in value if isinstance(v, (str, int, float))]
            if items:
                lines.append(f"{key}: {', '.join(items)}")
        elif isinstance(value, dict):
            items = [f"{k}={v}" for k, v in value.items() if isinstance(v, (str, int, float))]
            if items:
                lines.append(f"{key}: {', '.join(items)}")
    return "\n".join(lines)


# =====================================================
# EMBEDDING MODELS
# =====================================================
text_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts, batch_size=64):
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).astype(np.float32)

device = "cpu"
clip_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(clip_name, use_fast=True)
clip_model.eval()

@torch.no_grad()
def embed_images(paths, batch_size=16):
    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)

print("Modelos de embeddings preparados.")


# =====================================================
# RESTAURANT DOCUMENTS
# =====================================================
article_docs = []
seen = set()          # para detectar restaurantes duplicados exactos
duplicates = []
for i, r in enumerate(restaurants):
    name = clean(r.get("name"))
    if not name:
        continue

    # Duplicado exacto = mismo nombre, ubicación, cocina y platos -> nos quedamos solo con el primero
    key = (name, clean(r.get("location")), clean(r.get("food_style")), tuple(r.get("signatures") or []))
    if key in seen:
        duplicates.append(f"restaurant_{i} ({name})")
        continue
    seen.add(key)

    signatures   = ", ".join(r.get("signatures") or [])
    shortcomings = ", ".join(r.get("shortcomings") or [])

    # Texto en lenguaje natural: es lo que se convierte en embedding y lo que leerá el LLM
    lines = [
        f"Restaurant: {name}",
        f"Type: {clean(r.get('type'))}",
        f"Cuisine: {clean(r.get('food_style'))}",
        f"Location: {clean(r.get('location'))}",
        f"Signature dishes: {signatures}" if signatures else "",
        f"Vibe: {clean(r.get('vibe'))}" if r.get("vibe") else "",
        f"Environment: {clean(r.get('environment'))}" if r.get("environment") else "",
        f"Shortcomings: {shortcomings}" if shortcomings else "",
        f"Rating: {r.get('rating')}/5" if r.get("rating") is not None else "",
        f"Price level: {r.get('price_range')}/4" if r.get("price_range") is not None else "",
    ]
    text = "\n".join(l for l in lines if l)

    article_docs.append(
        Document(
            page_content=text,
            metadata={
                "doc_id": f"restaurant_{i}",
                "name": name,
                "type": clean(r.get("type")),
                "cuisine": clean(r.get("food_style")),
                "location": clean(r.get("location")),
                # numéricos: permiten filtros tipo {"price_range": {"$lte": 2}} o {"rating": {"$gte": 4.5}}
                "rating": float(r.get("rating") or 0),
                "price_range": int(r.get("price_range") or 0),
                "source": "restaurant",
            },
        )
    )
print(f"Documentos de restaurantes: {len(article_docs)}  (duplicados eliminados: {len(duplicates)} -> {duplicates})")
print("\n--- Ejemplo de documento de restaurante ---\n" + article_docs[0].page_content + "\n")


# =====================================================
# IMAGE DOCUMENTS  (foto <-> receta por id, NO por posición)
# =====================================================
img_by_id = {}
for p in images_paths:
    m = re.search(r"recipe(\d+)\.png$", os.path.basename(p))
    if m:
        img_by_id[int(m.group(1))] = p

image_docs = []
missing = []
for rec in recipes:
    rid = rec.get("id")
    try:
        rid = int(rid)
    except (TypeError, ValueError):
        missing.append(rid)
        continue

    p = img_by_id.get(rid)
    if p is None:
        missing.append(rid)
        continue

    # Nombre, cocina e ingredientes (sin pasos ni tiempos: es contexto para el LLM, no hace falta tanto)
    ingredients = ", ".join(str(x) for x in (rec.get("ingredients") or []))
    lines = [
        f"Dish: {clean(rec.get('name'))}",
        f"Cuisine: {clean(rec.get('cuisine'))}",
        f"Ingredients: {ingredients}" if ingredients else "",
        record_to_text(rec, skip={"id", "name", "cuisine", "ingredients", "directions",
                                  "servings", "prep_time", "cook_time", "total_time"}),
    ]
    text = "\n".join(l for l in lines if l).strip()

    image_docs.append(
        Document(
            page_content=text,
            metadata={
                "doc_id": f"img_{rid}",
                "image_path": p,
                "source": "recipe_image",
                "recipe_id": rid,
                "name": clean(rec.get("name")),
                "cuisine": clean(rec.get("cuisine")),
            },
        )
    )

print(f"Documentos de imágenes: {len(image_docs)}  (recetas sin foto: {len(missing)})")

# Comprobación del emparejamiento: recipe_id N tiene que ir con recipeN.png
for d in image_docs[:3]:
    print(f"  check -> recipe_id={d.metadata['recipe_id']}  file={os.path.basename(d.metadata['image_path'])}  name={d.metadata['name']}")


# =====================================================
# BUILD VECTOR DB
# =====================================================
if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)   # reset

A = embed_texts([d.page_content for d in article_docs])
article_db = Chroma(collection_name=ARTICLE_COLLECTION, persist_directory=DB_DIR)
article_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in article_docs],
    embeddings=A.tolist(),
    documents=[d.page_content for d in article_docs],
    metadatas=[d.metadata for d in article_docs],
)
print(f"Colección '{ARTICLE_COLLECTION}': {article_db._collection.count()} vectores")

V = embed_images([d.metadata["image_path"] for d in image_docs])
image_db = Chroma(collection_name=IMAGE_COLLECTION, persist_directory=DB_DIR)
image_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in image_docs],
    embeddings=V.tolist(),
    documents=[d.page_content for d in image_docs],
    metadatas=[d.metadata for d in image_docs],
)
print(f"Colección '{IMAGE_COLLECTION}': {image_db._collection.count()} vectores")

print(f"\nIndexación multimodal completada en {DB_DIR}")