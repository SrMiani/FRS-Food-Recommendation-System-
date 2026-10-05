
# Standard library
import glob
import json
import os
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


#Prepare image data
BASE_DIR=Path(__file__).resolve().parent
IMG_DIR = BASE_DIR.parent / "synthetic_recipe_images"

images_paths = sorted(glob.glob(str(IMG_DIR / "**" / "*.png"), recursive=True))


if not images_paths:
    raise FileNotFoundError(f"""No se encontraron imágenes en la carpeta {IMG_DIR} 
                            Asegúrate de que las imágenes estén presentes.""")

print (f"Se encontraron {len(images_paths)} imágenes en la carpeta {IMG_DIR}.")


#Load data 

with open("structured_restaurant_data.json", "r", encoding="utf-8") as f:
    restaurants = json.load(f)


with open("Recipes_with_image_descriptions.json","r",encoding="utf-8") as f:
    recipes=json.load(f)


print (f"Se encontraron {len(restaurants)} restaurantes y {len(recipes)} recetas en los archivos JSON.")


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


#Build document sets
article_docs=[]

for i, r in enumerate(restaurants):
    name=str(r.get("name","")).strip()
    if not name:
        continue

    text = (

    f"Restaurant: {name}\n"
    f"Cuisine: {r.get('food_style','')}\n"
    f"Location: {r.get('location','')}"
    )

    doc_id=f"restaurant_{i}"

    article_docs.append(
       Document(
           page_content=text.strip(), 
           metadata={
               "doc_id": doc_id, 
               "cuisine": r.get("food_style"),
               "location":r.get("location"),
               "source":"restaurant"
               },
        )
    )

print (f"Se han creado {len(article_docs)} documentos de restaurantes.")


#Build Images documents

image_docs=[]

for i, (p, rec) in enumerate(zip(images_paths, recipes)):
    doc_id = f"img_{i}"

    image_docs.append(
        Document(
            page_content=rec.get("name", f"recipe image {i}"),
            metadata={
                "doc_id": doc_id,
                "image_path": p,
                "source": "recipe_image",
                "recipe_id": rec.get("id"),
                "cuisine": rec.get("cuisine"),
            },
        )
    )

print (f"Se han creado {len(image_docs)} documentos de imágenes de recetas.")


#DB_DIR = str((Path.home() / "chroma_multimodal").resolve())
PROJECT_ROOT = BASE_DIR.parents[1]                 # .../FRS
DB_DIR       = str(PROJECT_ROOT / "chroma_multimodal")

if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR) #Resetear vector DB 

A = embed_text([d.page_content for d in article_docs])

article_db =Chroma(
    collection_name="restaurant_articles",
    persist_directory=DB_DIR,
)


article_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in article_docs],
    embeddings=A.tolist(),
    documents = [d.page_content for d in article_docs],
    metadatas=[d.metadata for d in article_docs],
)


print(f"Se ha creado la base de datos de artículos de restaurantes en {DB_DIR}.")


#IMAGE db

V = embed_images([d.metadata["image_path"] for d in image_docs])

image_db = Chroma(
    
    collection_name="recipe_images",
    persist_directory=DB_DIR,
)

image_db._collection.upsert(
    ids=[d.metadata["doc_id"] for d in image_docs],
    embeddings=V.tolist(),
    documents=[d.page_content for d in image_docs],
    metadatas=[d.metadata for d in image_docs],
)

print(f"Se ha creado la base de datos de imágenes de recetas en {DB_DIR}.")
print("Proceso de indexación multimodal completado exitosamente.")
