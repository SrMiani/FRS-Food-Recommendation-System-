FRS - Food Recommendation System (Plate Picker)

Que es?
-Recomienda restaurantes y recetas utilizando RAG multimodal (texto e imagen) y un sistema multi-agente con LangGraph

Que hace?
Introduces los hábitos alimentarios de una persona (historial de restaurantes, publicaciones, restricciones…) y el sistema devuelve 5 restaurantes y 5 recetas que existen en la base de datos 
y encajan con sus gustos y su perfil alimentario, explicando el por qué de esa decisión.


Evaluación de resultados

Métrica    Valor
hit@5       0.89
MRR         0.74
latencia    18.7ms


Estructura del Sistema

Embedding de texto -  all-MiniLM-L6-V2 - Convierte cada restaurante en un vector
Embedding de imagen - CLIP(512d) - Para buscar fotos de recetas escribiendo texto
VectorDB - ChromaDB- Para buscar por similitud vectorial
Agentes  - LangGraph - Para coordinar los agentes con un state compartido.
LLM - OpenAI(gpt-5-mini) - Perfil, análisis y síntesis


<img width="755" height="780" alt="image" src="https://github.com/user-attachments/assets/c4a59f9d-74d6-4f0b-adea-f394311dcdbe" />




