# 🍽️ FRS - Food Recommendation System (Plate Picker)

<p align="center">
  <a href="https://platepicker-242987336120.europe-west1.run.app/">
    <img src="https://img.shields.io/badge/Demo%20en%20vivo-Probar%20ahora-brightgreen?style=for-the-badge" alt="Demo en vivo" />
  </a>
</p>

## Que es?

Recomienda restaurantes y recetas utilizando **RAG multimodal** (texto e imagen) y un **sistema multi-agente con LangGraph**.

## Que hace?

Introduces los hábitos alimentarios de una persona (historial de restaurantes, publicaciones, restricciones…) y el sistema devuelve **5 restaurantes y 5 recetas que existen en la base de datos** y encajan con sus gustos y su perfil alimentario, explicando el por qué de esa decisión.

---

## 📊 Evaluación de resultados

| Métrica | Valor |
|:--|:--:|
| hit@5 | **0.89** |
| MRR | **0.74** |
| latencia | **18.7ms** |

---

## 🗺️ Cómo funciona

```mermaid
flowchart TB
    U(["👤 Usuario<br/>describe cómo come"]):::user
    UI["🖥️ Interfaz Gradio + FastAPI<br/>Docker · Google Cloud Run"]:::ui
    P["① Agente Perfil<br/>gustos · restricciones · hábitos"]:::agent
    T["② Tool de búsqueda · RAG<br/>20 restaurantes + 20 recetas reales"]:::tool
    A1["③ 📈 Tendencias"]:::agent
    A2["③ 🎨 Estilos"]:::agent
    A3["③ 🥗 Nutrición"]:::agent
    S["④ Agente Síntesis<br/>elige top 5 + top 5 y copia el id"]:::agent
    V{"⑤ Validación<br/>¿el id existe?"}:::check
    R(["🍽️ 5 restaurantes + 5 recetas<br/>con su explicación"]):::user

    subgraph RAG["🔎 Retrieval multimodal"]
        E1["MiniLM · texto → 384d"]:::emb --> DB1[("ChromaDB<br/>207 restaurantes")]:::db
        E2["CLIP · texto → 512d"]:::emb --> DB2[("ChromaDB<br/>109 fotos de recetas")]:::db
    end

    LLM["🤖 OpenAI gpt-5-mini<br/>JSON mode"]:::llm

    U --> UI --> P --> T
    T <--> RAG
    T --> A1 & A2 & A3
    A1 & A2 & A3 --> S --> V
    V -- "✅ sí" --> R
    V -. "❌ no: se descarta" .-> S
    P & S -.- LLM

    classDef user fill:#4a0080,stroke:#b366ff,color:#fff
    classDef ui fill:#1f1f2e,stroke:#b366ff,color:#fff
    classDef agent fill:#2d1b4e,stroke:#b366ff,color:#fff
    classDef tool fill:#123a5c,stroke:#4da3ff,color:#fff
    classDef emb fill:#123a5c,stroke:#4da3ff,color:#fff
    classDef db fill:#0e3b2e,stroke:#3ddc97,color:#fff
    classDef check fill:#5c3b12,stroke:#ffb84d,color:#fff
    classDef llm fill:#3b3b3b,stroke:#aaaaaa,color:#fff
    style RAG fill:#0d1117,stroke:#4da3ff,color:#fff
```

### 📚 Conceptos clave

| Concepto | Qué es | Dónde se usa en FRS |
|:--|:--|:--|
| **Embedding** | Convertir un texto o una imagen en una lista de números que representa su significado | MiniLM (texto, 384d) y CLIP (texto ↔ imagen, 512d) |
| **Vector DB** | Base de datos que encuentra "lo más parecido a esto" comparando vectores | ChromaDB, 2 colecciones |
| **RAG** | El LLM responde con datos reales que se buscan antes, no con lo que "recuerda" | Paso ② da al agente candidatos reales |
| **Multimodal** | Buscar en texto e imágenes a la vez | "pizza con albahaca" encuentra la foto de una margarita |
| **Agente** | Un LLM con un rol, un objetivo y un contexto concretos | Perfil, Tendencias, Estilos, Nutrición, Síntesis |
| **State** | La "libreta" compartida que cada agente lee y completa | `WorkflowState` de LangGraph |
| **Fan-out / fan-in** | Lanzar varios agentes a la vez y juntar sus resultados | Paso ③ en paralelo → paso ④ |
| **Tool** | Una función real que el agente usa en vez de inventarse la respuesta | Búsqueda en ChromaDB |
| **Validación** | Comprobar en código lo que dice el LLM | `keep_valid()` descarta ids inventados → 0 alucinaciones |
| **JSON mode** | Obligar al LLM a responder en JSON válido | Evita respuestas rotas |
| **Docker + Cloud Run** | Empaquetar la app en un contenedor y ejecutarla en la nube bajo demanda | Demo pública |

---

## 🧩 Estructura del Sistema

| Componente | Tecnología | Función |
|:--|:--|:--|
| Embedding de texto | `all-MiniLM-L6-V2` | Convierte cada restaurante en un vector |
| Embedding de imagen | `CLIP` (512d) | Para buscar fotos de recetas escribiendo texto |
| VectorDB | `ChromaDB` | Para buscar por similitud vectorial |
| Agentes | `LangGraph` | Para coordinar los agentes con un state compartido |
| LLM | `OpenAI` (gpt-5-mini) | Perfil, análisis y síntesis |
| API | `FastAPI` | Endpoint `/recommend` |
| Interfaz | `Gradio` | Demo web con la estética de Loading SSG |
| Despliegue | `Docker` + `Google Cloud Run` | Demo pública en la nube |



<p align="center">
  <img width="755" height="780" alt="image" src="https://github.com/user-attachments/assets/c4a59f9d-74d6-4f0b-adea-f394311dcdbe" />
</p>


---

## 🚀 Pruébalo

<p align="center">
  <a href="https://platepicker-242987336120.europe-west1.run.app/">
    <img src="https://img.shields.io/badge/Probar%20la%20demo-Abrir-brightgreen?style=for-the-badge" alt="Probar la demo" />
  </a>
</p>

> ⏱️ La primera visita puede tardar ~30 s en arrancar (el servidor se apaga cuando no se usa). Cada recomendación tarda ~30 s: trabajan 5 agentes.

---

## ✅ Hecho / 🔜 Siguiente

- [x] Retrieval multimodal evaluado con golden set (hit@5 0.89, MRR 0.74)
- [x] Sin alucinaciones: tool real contra ChromaDB + validación de ids (10/10 recomendaciones reales)
- [x] API con FastAPI
- [x] Interfaz web con Gradio
- [x] Despliegue en Google Cloud Run con Docker
- [ ] Servidor MCP para usar FRS desde cualquier asistente de IA
- [ ] Restricciones alimentarias como filtro obligatorio
- [ ] Embeddings multilingües (mejorar las consultas en español)
