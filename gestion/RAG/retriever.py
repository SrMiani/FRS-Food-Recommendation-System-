import os
from pathlib import Path

from httpx2 import query
from networkx import display
import numpy as np
import torch
from PIL import Image
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor


BASE_DIR=Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]                 # .../FRS
DB_DIR       = str(PROJECT_ROOT / "chroma_multimodal")



if not os.path.exists(DB_DIR):
    raise RuntimeError(f"DB_DIR '{DB_DIR}' does not exist. Please run 'multimodal_index.py' first to create the vector database.")


article_db=Chroma(
    collection_name="restaurant_articles",
    persist_directory=DB_DIR,
)

image_db=Chroma(
    collection_name="recipe_images",
    persist_directory=DB_DIR,
)

n_articles= article_db._collection.count()
n_images= image_db._collection.count()

if n_articles <=0 or n_images <=0:
    raise RuntimeError(
        "One or more collections are empty."

    )

print (f"Loaded {n_articles} restaurant articles and {n_images} recipe images from the vector database ")


#Embbedding models

#Iniciar embeddings model
text_model = SentenceTransformer('all-MiniLM-L6-v2')

#Text Embedding function (384d)
def embed_text(texts,batch_size=64):
    return text_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True, 
    ).astype(np.float32)

print ("Embedding de texto preparado.")

#Image embedding function(512d)
device="cpu"
clip_name="openai/clip-vit-base-patch32"
clip_model=CLIPModel.from_pretrained(clip_name).to(device)
clip_processor=CLIPProcessor.from_pretrained(clip_name,use_fast=True)
clip_model.eval()


@torch.no_grad()
def embed_images(paths, batch_size=16):
    vecs = []
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = clip_processor(images=imgs, return_tensors="pt").to(device)
        feats = clip_model.get_image_features(**inputs)          # (B,512)
        feats = feats / feats.norm(dim=-1, keepdim=True)         # cosine-ready
        vecs.append(feats.cpu().numpy().astype(np.float32))
    return np.vstack(vecs)

print(" Image embedder preparado.")

@torch.no_grad()
def embed_query_clip_text(query: str):
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(device)
    feats = clip_model.get_text_features(**inputs)              # (1,512)
    feats = feats / feats.norm(dim=-1, keepdim=True)            # cosine-ready
    return feats[0].cpu().numpy().astype(np.float32)


print(" CLIP embedder preparado.")


# ================================
# Retrieval utilities
# ================================

# Chroma returns lists-of-lists; unwrap the first query.
def _unwrap(res: dict):  
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return ids, docs, metas, dists

def _to_similarity(dists):
    """Convert 'smaller is better' distance to 'larger is better' similarity."""
    d = np.array(dists, dtype=np.float32)
    return 1.0 - d

def _minmax(x):
    """Min-max normalize to [0, 1] with safe handling for constant arrays."""
    x = np.array(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if abs(hi - lo) < 1e-8:
        return np.ones_like(x)  # all equal -> treat as same confidence
    return (x - lo) / (hi - lo)


def print_hits(ids, docs, metas, dists, title: str, max_chars: int = 180):
    print(f"\n=== {title} ===")
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        dist = float(dists[i]) if i < len(dists) else None

        snippet = (docs[i] or "").replace("\n", " ").strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."

        # compact metadata view
        cuisine = meta.get("cuisine", "N/A") if isinstance(meta, dict) else "N/A"
        location = meta.get("location", "N/A") if isinstance(meta, dict) else "N/A"
        doc_id = meta.get("doc_id", "N/A") if isinstance(meta, dict) else "N/A"
        source = meta.get("source", "N/A") if isinstance(meta, dict) else "N/A"

        print(f"[{i+1}] id={doc_id} | cuisine={cuisine} | location={location} | source={source} | distance={dist:.4f}")
        print(f"{snippet}")


    # ================================
# Article retrieval
# ================================

# Similarity retrieval over restaurant articles with optional metadata filtering.
def retrieve_articles(query: str, k: int = 5, where: dict | None = None):

    q_vec = embed_text([query])[0]  # 384-d, cosine-ready

    res = article_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs, metas, dists = _unwrap(res)
    sims = _to_similarity(dists)
    return ids, docs, metas, sims

print("✅ Article retrieval ready")    

# ================================
# Image retrieval
# ================================

def retrieve_images_by_image(query_image_path: str, k: int = 5, where: dict | None = None):

    q_vec = embed_images([query_image_path])[0]  # 512-d, cosine-ready

    res = image_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return _unwrap(res)

print("✅ Image retrieval ready")


# Similarity retrieval over food images using an image query.
def retrieve_images_by_text(query: str, k: int = 5, where: dict | None = None):

    q_vec = embed_query_clip_text(query)  # 512-d  

    res = image_db._collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs, metas, dists = _unwrap(res)
    sims = _to_similarity(dists)
    return ids, docs, metas, sims

print("✅ Image retrieval ready")



# ================================
# Demo 1 — Article similarity search (no filter)
# ================================

q = "cozy restaurant with noodles and warm atmosphere"

ids, docs, metas, dists = retrieve_articles(q, k=5, where=None)
print_hits(ids, docs, metas, dists, title="Demo 1 — Article similarity search (no filter)")

print("✅ Demo 1 complete")




# ================================
# Demo 2 — Article similarity search + metadata filter
# ================================

q = "handmade pasta and romantic dinner"

# ---- metadata constraint (must exist in your dataset) ----
where_filter = {"location": "Pasadena"}  # adjust if needed

ids, docs, metas, dists = retrieve_articles(q, k=5, where=where_filter)

if len(ids) == 0:
    print("⚠️ No results found with current filter.")
else:
    print_hits(ids, docs, metas, dists, title="Demo 2 — Article similarity search + metadata filter")
    
print("✅ Demo 2 complete")



# ================================
# Demo 3 — Image similarity search (image→image)
# ================================

meta_all = image_db._collection.get(include=["metadatas"])["metadatas"]

QUERY_INDEX = 0  # ← change this (0 … N-1) to try different images

if QUERY_INDEX >= len(meta_all):
    raise ValueError("QUERY_INDEX out of range.")

query_img = meta_all[QUERY_INDEX]["image_path"]

print(f"Query image: {query_img}")
img = Image.open(query_img)
img.thumbnail((300, 300))
img.show()
# display(Image.open(query_img))

# TODO:
# 1. Create an optional metadata filter for recipe images
# 2. Retrieve the top-5 most similar images using query_img
# 3. Display the retrieved results with metadata (use title="Demo 3 — Image similarity search (image→image)" when printing results)

where_filter = {"cuisine": "Italian"}  # adjust if needed
ids, docs, metas, dists = retrieve_images_by_text(query_img, k=5, where=where_filter)
print_hits(ids, docs, metas, dists, title="Demo 3 — Image similarity search (image→image)" )


print("✅ Demo 3 complete")
print("🎉 Similarity Retrieval with Metadata Filtering COMPLETE")



# =====================================================
# MULTIMODAL FUSION RETRIEVAL
# =====================================================


#Fuse rank

def fuse_rank(
    query:str,
    k_text: int = 5,
    k_img: int = 5,
    w_text: float = 0.6,
    w_img: float = 0.4,
    where_text: dict | None=None,
    where_img: dict | None=None,
    top_n: int = 5
):

    #retrieve por modalidad
    t_ids,t_docs,t_metas,t_sims = retrieve_articles(query, k=k_text, where=where_text)
    i_ids,i_docs,i_metas,i_sims = retrieve_images_by_text(query, k=k_img, where=where_img)


    #normalizar resultados
    t_norm=_minmax(t_sims)
    i_norm=_minmax(i_sims)


    #Construir lista con las puntuaciones mixtas de texto e imagen

    rows=[]
    for i in range(len(t_ids)):
        rows.append({
            "modality":"article",
            "id":t_metas[i].get("doc_id",t_ids[i]) if isinstance(t_metas[i],dict) else t_ids[i],
            "cuisine": t_metas[i].get("cuisine", "N/A") if isinstance(t_metas[i], dict) else "N/A",
            "location": t_metas[i].get("location", "N/A") if isinstance(t_metas[i], dict) else "N/A",
            "source": t_metas[i].get("source", "N/A") if isinstance(t_metas[i], dict) else "N/A",
            "text_score": float(t_norm[i]),
            "img_score": 0.0,
            "fused": float(w_text * t_norm[i]),
            "snippet": (t_docs[i] or "").replace("\n", " ").strip(),
        })

    for j in range(len(i_ids)):
        rows.append({
            "modality": "image",
            "id": i_metas[j].get("doc_id", i_ids[j]) if isinstance(i_metas[j], dict) else i_ids[j],
            "cuisine": i_metas[j].get("cuisine", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "location": i_metas[j].get("location", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "source": i_metas[j].get("source", "N/A") if isinstance(i_metas[j], dict) else "N/A",
            "text_score": 0.0,
            "img_score": float(i_norm[j]),
            "fused": float(w_img * i_norm[j]),
            "snippet": (i_docs[j] or "").replace("\n", " ").strip(),
        })

    

# Sort by fused score (desc rerank)
    rows.sort(key=lambda r: r["fused"], reverse=True)
    
    # if top_n not specified, return full pool (k_text + k_img)
    if top_n is None:
        return rows

    top_n = max(0, min(int(top_n), len(rows)))
    return rows[:top_n]



def print_fused(rows, title: str, max_chars: int = 90):
    print(f"\n=== {title} ===")
    for idx, r in enumerate(rows, start=1):
        snippet = r["snippet"]
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        print(
            f"[{idx}] {r['modality']} | id={r['id']} | cuisine={r['cuisine']} | "
            f"location={r['location']} | fused={r['fused']:.4f} "
            f"(text={r['text_score']:.4f}, img={r['img_score']:.4f})"
        )
        print(snippet)



# ================================
# Demo 3 — Weight tuning
# ================================

q = "fresh sushi and minimalist presentation"

# TODO:
# 1. Run fusion ranking with a text-heavy setting and print results (use title="Demo 3A — Text-heavy fusion (w_text=0.8, w_img=0.2)")
# 2. Run fusion ranking with an image-heavy setting and print results (use title="Demo 3B — Image-heavy fusion (w_text=0.3, w_img=0.7)")
# 3. Select 5 results from each modality, but only show the top 5 fused results

rows = fuse_rank(
    q,
    k_text=5,
    k_img=5,
    w_text=0.8,
    w_img=0.2,
    where_text=None,
    where_img=None,
    top_n=5 
)

rows2 = fuse_rank(
    q,
    k_text=5,
    k_img=5,
    w_text=0.3,
    w_img=0.7,
    where_text=None,
    where_img=None,
    top_n=5 
)

print_fused(rows,title="Demo 3A- Text-heavy fusion (w_text=0.8,w_img=0.2)")
print_fused(rows2,title="Demo 3B- Image-heavy fusion (w_text=0.3,w_img=0.7)")


print("✅ Demo 3 complete")
print("🎉 Multimodal Similarity Fusion and Retrieval Ranking COMPLETE")


