from pathlib import Path
from collections import Counter
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
DB_DIR = str(PROJECT_ROOT / "chroma_multimodal")

article_db = Chroma(collection_name="restaurant_articles", persist_directory=DB_DIR)
image_db   = Chroma(collection_name="recipe_images",         persist_directory=DB_DIR)

arts = article_db._collection.get(include=["documents", "metadatas"])
imgs = image_db._collection.get(include=["documents", "metadatas"])

print("Restaurantes:", len(arts["ids"]))
print("Imágenes:", len(imgs["ids"]))
print("Cocinas:", Counter(m.get("cuisine") for m in arts["metadatas"]))
print("Ubicaciones:", Counter(m.get("location") for m in arts["metadatas"]))

print("\n--- 3 restaurantes de ejemplo ---")
for i in range(3):
    print(arts["metadatas"][i])
    print(arts["documents"][i][:300], "\n")

print("--- 3 imágenes de ejemplo ---")
for i in range(min(3, len(imgs["ids"]))):
    print(imgs["metadatas"][i])
    print(imgs["documents"][i][:200], "\n")