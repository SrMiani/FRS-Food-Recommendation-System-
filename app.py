"""PlatePicker demo — Gradio UI con la estética de Loading SSG.

Va en la raíz del proyecto (al lado de la carpeta gestion/).
Local:  python app.py   ->  http://127.0.0.1:7860
"""
import base64
import html
import os
import sys
import time
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gestion" / "agents"))
from workflow import run_workflow  # noqa: E402

# ---------- protección de la API key ----------
MAX_PER_HOUR = int(os.getenv("MAX_PER_HOUR", "20"))
_calls: list[float] = []


def _rate_ok() -> bool:
    now = time.time()
    _calls[:] = [t for t in _calls if now - t < 3600]
    if len(_calls) >= MAX_PER_HOUR:
        return False
    _calls.append(now)
    return True


# ---------- helpers ----------
EXAMPLES = [
    "Últimamente como mucho ramen y sushi, me gustan los sitios tranquilos y minimalistas. "
    "Soy vegetariano y no me importa pagar un poco más si la calidad es buena.",
    "I love spicy food, tacos and Korean BBQ. Lively places with music. Budget-friendly, no seafood.",
    "Healthy eating: salads, grilled fish, Mediterranean food. Gluten-free. Quiet place with a terrace.",
]

_img_cache: dict[str, str] = {}


def _find_image(path: str | None) -> str | None:
    """Devuelve la imagen como data-URI. Acepta rutas absolutas de otro PC buscando por nombre."""
    if not path:
        return None
    if path in _img_cache:
        return _img_cache[path]
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        name = Path(path.replace("\\", "/")).name
        hits = list(ROOT.rglob(name))
        if not hits:
            return None
        p = hits[0]
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    uri = f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
    _img_cache[path] = uri
    return uri


def _e(x) -> str:
    return html.escape(str(x)) if x not in (None, "") else ""


def _split_recs(recs):
    """Acepta {'restaurants': [...], 'recipes': [...]} o variantes de nombre."""
    if not isinstance(recs, dict):
        return [], []
    rest = recs.get("restaurants") or recs.get("top_restaurants") or []
    reci = recs.get("recipes") or recs.get("top_recipes") or []
    return rest, reci


def profile_html(profile) -> str:
    if isinstance(profile, dict):
        rows = []
        for k, v in profile.items():
            if isinstance(v, list):
                v = ", ".join(map(str, v))
            elif isinstance(v, dict):
                v = ", ".join(f"{a}: {b}" for a, b in v.items())
            rows.append(f"<div class='kv'><span>{_e(k.replace('_', ' '))}</span><p>{_e(v)}</p></div>")
        body = "".join(rows)
    else:
        body = f"<p>{_e(profile)}</p>"
    return f"<div class='card profile'><div class='eyebrow'>01 · Perfil detectado</div>{body}</div>"


def restaurants_html(items, by_id) -> str:
    cards = []
    for i, r in enumerate(items, 1):
        meta = by_id.get(r.get("id"), {})
        name = r.get("name") or meta.get("name", "")
        price = meta.get("price_range")
        price = "€" * int(price) if str(price).isdigit() else _e(price)
        tags = " · ".join(_e(x) for x in [meta.get("cuisine"), meta.get("location"), price] if x)
        rating = meta.get("rating")
        cards.append(f"""
        <div class='card rec'>
          <div class='rank'>{i:02d}</div>
          <div class='rec-body'>
            <div class='rec-head'><h3>{_e(name)}</h3>{f"<span class='pill'>★ {_e(rating)}</span>" if rating else ""}</div>
            <div class='tags'>{tags}</div>
            <p>{_e(r.get('reasoning', ''))}</p>
          </div>
        </div>""")
    return "<div class='eyebrow'>02 · Restaurantes</div>" + "".join(cards)


def recipes_html(items, by_id) -> str:
    cards = []
    for r in items:
        meta = by_id.get(r.get("id"), {})
        img = _find_image(meta.get("image_path"))
        img_tag = f"<img src='{img}' alt=''>" if img else "<div class='noimg'>🍽️</div>"
        cards.append(f"""
        <div class='card recipe'>
          {img_tag}
          <div class='recipe-body'>
            <h3>{_e(r.get('name') or meta.get('name', ''))}</h3>
            <div class='tags'>{_e(meta.get('cuisine', ''))}</div>
            <p>{_e(r.get('reasoning', ''))}</p>
          </div>
        </div>""")
    return "<div class='eyebrow'>03 · Recetas</div><div class='grid'>" + "".join(cards) + "</div>"


def recommend(text: str):
    text = (text or "").strip()
    if len(text) < 15:
        raise gr.Error("Cuéntame un poco más sobre cómo comes (mínimo 15 caracteres).")
    if len(text) > 1500:
        raise gr.Error("Máximo 1500 caracteres.")
    if not _rate_ok():
        raise gr.Error("La demo ha llegado al límite de uso por hora. Vuelve en un rato.")

    t0 = time.time()
    try:
        state = run_workflow(text)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Algo ha fallado en el sistema de agentes: {exc}") from exc

    recs = state.get("final_recommendations", {})
    if isinstance(recs, dict) and "error" in recs:
        raise gr.Error("El LLM no ha devuelto una respuesta válida. Prueba otra vez.")

    rest, reci = _split_recs(recs)
    rest_by_id = {c.get("doc_id"): c for c in state.get("retrieved_restaurants", [])}
    reci_by_id = {c.get("doc_id"): c for c in state.get("retrieved_recipes", [])}

    secs = time.time() - t0
    stats = (f"<div class='stats'><span>{len(rest)}+{len(reci)} recomendaciones</span>"
             f"<span>100% validadas contra la base de datos</span><span>{secs:.1f} s</span></div>")
    return (
        profile_html(state.get("user_profile", "")),
        stats + restaurants_html(rest, rest_by_id),
        recipes_html(reci, reci_by_id),
    )


# ---------- estilo Loading ----------
CSS = """
:root, .dark { --sky:#050507; --purple:#4a0080; --flash:#b366ff; --glow:rgba(120,0,180,.6);
  --card:rgba(255,255,255,.025); --line:rgba(255,255,255,.08); --mut:rgba(255,255,255,.62); }
body, .gradio-container { background:var(--sky) !important; color:#fff !important;
  font-family:Inter, system-ui, sans-serif !important; }
.gradio-container { max-width:1100px !important; margin:auto !important; }
footer { display:none !important; }
body { background:radial-gradient(900px 500px at 10% -10%, rgba(74,0,128,.28), transparent 70%), #050507 !important; }
.gradio-container { background:transparent !important; }
.hero { padding:40px 0 18px; }
.hero .top { font-family:'JetBrains Mono',monospace; font-size:12px; letter-spacing:.16em; color:var(--mut);
  text-transform:uppercase; display:flex; justify-content:space-between; }
.hero .top b { color:#fff; font-weight:500 } .hero .top b i { color:var(--flash); font-style:normal }
.hero h1 { font-size:clamp(36px,6vw,64px); font-weight:900; letter-spacing:-.03em; margin:22px 0 8px; line-height:1 }
.hero h1 span { color:var(--flash) }
.hero p { color:var(--mut); font-size:17px; max-width:44em; margin:0 }
.pipeline { display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; font-family:'JetBrains Mono',monospace; font-size:12px }
.pipeline span { border:1px solid var(--line); border-radius:999px; padding:5px 11px; color:var(--mut) }
.pipeline span.arr { border:none; padding:5px 0; color:var(--flash) }
.pipeline span.on { border-color:var(--glow); color:#fff; box-shadow:0 0 18px -6px var(--glow) }
.eyebrow { font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--flash); letter-spacing:.14em;
  text-transform:uppercase; margin:22px 0 10px }
.card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px 18px; margin-bottom:12px }
.card h3 { margin:0; font-size:17px; font-weight:700; color:#fff }
.card p { color:#e4e4ea; margin:6px 0 0; font-size:14.5px; line-height:1.5 }
.tags { font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--mut); margin-top:4px }
.profile .kv { display:grid; grid-template-columns:150px 1fr; gap:12px; padding:6px 0; border-top:1px solid var(--line) }
.profile .kv:first-of-type { border-top:none }
.profile .kv span { font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--flash); text-transform:uppercase; padding-top:2px }
.profile .kv p { margin:0 }
.rec { display:flex; gap:16px; align-items:flex-start }
.rank { font-family:'JetBrains Mono',monospace; font-size:22px; color:var(--flash); min-width:34px }
.rec-body { flex:1 } .rec-head { display:flex; justify-content:space-between; gap:10px; align-items:center }
.pill { font-family:'JetBrains Mono',monospace; font-size:12px; border:1px solid var(--glow); border-radius:999px; padding:2px 9px; white-space:nowrap }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:14px }
.recipe { padding:0; overflow:hidden; margin:0 }
.recipe img, .recipe .noimg { width:100%; aspect-ratio:4/3; object-fit:cover; display:block; background:#15151d }
.recipe .noimg { display:flex; align-items:center; justify-content:center; font-size:40px }
.recipe-body { padding:12px 14px 14px }
.stats { display:flex; flex-wrap:wrap; gap:8px; margin-top:6px }
.stats span { font-family:'JetBrains Mono',monospace; font-size:12px; color:#fff; background:rgba(74,0,128,.35);
  border:1px solid var(--glow); border-radius:999px; padding:4px 10px }
#go { background:linear-gradient(90deg,#33005a,#5c14a3) !important; border:none !important; color:#fff !important;
  font-weight:700 !important; box-shadow:0 0 30px -8px var(--glow) !important }
textarea { background:var(--card) !important; color:#fff !important; border:1px solid var(--line) !important; font-size:15px !important }
.foot { font-family:'JetBrains Mono',monospace; font-size:12px; color:var(--mut); border-top:1px solid var(--line);
  margin-top:28px; padding:14px 0; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px }
.foot a { color:var(--flash) !important; text-decoration:none }
@media (max-width:600px){ .hero .top span { display:none } .profile .kv { grid-template-columns:1fr; gap:2px } }
"""

HEAD = """<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">"""

HERO = """
<div class='hero'>
  <div class='top'><b>LOADING <i>SSG</i></b><span>Loading Software · Proyecto 001</span></div>
  <h1>Plate<span>Picker</span></h1>
  <p>Cuéntale cómo comes y un equipo de agentes de IA te recomienda restaurantes y recetas que existen de verdad.</p>
  <div class='pipeline'>
    <span class='on'>Perfil</span><span class='arr'>→</span><span class='on'>RAG multimodal · ChromaDB</span><span class='arr'>→</span>
    <span class='on'>Tendencias ∥ Estilos ∥ Nutrición</span><span class='arr'>→</span><span class='on'>Síntesis</span><span class='arr'>→</span>
    <span class='on'>Validación de ids</span>
  </div>
</div>"""

FOOT = """
<div class='foot'>
  <span>MiniLM + CLIP · ChromaDB · LangGraph · OpenAI · hit@5 0.89 · MRR 0.74</span>
  <span><a href='https://loadingssg.com'>loadingssg.com</a> · <a href='https://github.com/SrMiani/FRS-Food-Recommendation-System-'>GitHub</a></span>
</div>"""

theme = gr.themes.Base(primary_hue="purple", neutral_hue="zinc").set(
    body_background_fill="#050507", body_background_fill_dark="#050507",
    block_background_fill="#050507", block_background_fill_dark="#050507",
    block_border_width="0px", body_text_color="#ffffff", body_text_color_dark="#ffffff",
)

_G6 = int(gr.__version__.split(".")[0]) >= 6
_style = dict(css=CSS, head=HEAD, theme=theme)

with gr.Blocks(title="PlatePicker · Loading SSG", **({} if _G6 else _style)) as demo:
    gr.HTML(HERO)
    inp = gr.Textbox(lines=4, show_label=False, placeholder="Ej.: me encanta la comida picante, como mucho tacos... soy vegetariano")
    gr.Examples(EXAMPLES, inputs=inp, label="Prueba con un ejemplo")
    btn = gr.Button("Recomiéndame", elem_id="go", size="lg")
    gr.HTML("<div class='tags' style='margin-top:6px'>Tarda unos 30 s: trabajan 5 agentes. Los datos son sintéticos (capstone de IBM). Mejor en inglés.</div>")
    out_profile = gr.HTML()
    out_rest = gr.HTML()
    out_reci = gr.HTML()
    gr.HTML(FOOT)
    btn.click(recommend, inp, [out_profile, out_rest, out_reci], concurrency_limit=2)

if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")),
        **(_style if _G6 else {}),
    )
