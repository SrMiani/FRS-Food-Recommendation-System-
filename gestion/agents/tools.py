# tools.py — búsqueda real en Chroma (mismo índice que multimodal_index.py)
from pathlib import Path
import numpy as np
import torch
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]      # agents -> gestion -> FRS
DB_DIR = str(PROJECT_ROOT / "chroma_multimodal")

text_model = SentenceTransformer("all-MiniLM-L6-v2")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)

article_db = Chroma(collection_name="restaurant_articles", persist_directory=DB_DIR)
image_db = Chroma(collection_name="recipe_images", persist_directory=DB_DIR)


def _query(db, vector, k):
    res = db._collection.query(query_embeddings=[vector], n_results=k,
                               include=["documents", "metadatas", "distances"])
    results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        item = dict(meta)                       # doc_id, name, cuisine...
        item["text"] = doc
        item["similarity"] = round(1 - dist / 2, 3)
        results.append(item)
    return results


def search_restaurants(query: str, k: int = 20) -> list:
    vec = text_model.encode([query], normalize_embeddings=True)[0].astype(np.float32).tolist()
    return _query(article_db, vec, k)


@torch.no_grad()
def search_recipes(query: str, k: int = 20) -> list:
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    feats = clip_model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return _query(image_db, feats[0].numpy().astype(np.float32).tolist(), k)