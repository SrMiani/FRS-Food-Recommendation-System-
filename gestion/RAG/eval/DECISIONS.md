## 2026-10-12 — Baseline (all-MiniLM-L6-v2, k=5)
- hit@5 total: 0.89 · MRR: 0.74 · latencia: 19 ms
- Español: 0.25 → el modelo no entiende español (empareja por forma, no por significado)
- Umbral no-match: no separa "kosher deli" (0.51) de los aciertos → hará falta verificación con LLM
- Próximo experimento: modelo multilingüe