FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 HF_HOME=/models PIP_NO_CACHE_DIR=1
WORKDIR /app

# PyTorch solo CPU (mucho más ligero que la versión con CUDA)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt "gradio>=5,<6" "huggingface-hub<1.0"

# Descarga los modelos en la imagen para que no se bajen en cada arranque
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); \
from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

COPY . .

CMD ["python", "app.py"]
