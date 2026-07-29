"""
Optimized with the lightest & fastest models to run in Streamlit!
Run:      streamlit run app.py
Install:  pip install streamlit torch diffusers transformers accelerate safetensors pillow
"""

import base64
import io
import random
import time

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ────────────────────────────── page ──────────────────────────────
st.set_page_config(page_title="Atelier // Latent", page_icon="◉", layout="wide")

# ────────────────────────────── theme (pure presentation) ─────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,900;1,9..144,500&family=Manrope:wght@400;500;600;700&family=Spline+Sans+Mono:wght@400;500;600&display=swap');

.main .block-container { padding-top: 3.3rem; max-width: 1180px; }

.stApp {
    color:#d8cfd6;
    background:
        radial-gradient(1000px 600px at 90% -10%, rgba(240,169,59,.15), transparent 60%),
        radial-gradient(900px 700px at -10% 110%, rgba(95,198,192,.12), transparent 58%),
        radial-gradient(700px 500px at 50% 120%, rgba(217,140,140,.08), transparent 60%),
        #100c13;
}
.stApp::before {
    content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
    background-image:
        linear-gradient(rgba(244,237,225,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(244,237,225,.025) 1px, transparent 1px);
    background-size:46px 46px;
    -webkit-mask-image:radial-gradient(ellipse 92% 72% at 50% 16%, black, transparent 80%);
            mask-image:radial-gradient(ellipse 92% 72% at 50% 16%, black, transparent 80%);
}
.stApp::after {
    content:""; position:fixed; inset:-50%; pointer-events:none; z-index:0; opacity:.05;
    mix-blend-mode:overlay;
    background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%25' height='100%25' filter='url(%23n)'/></svg>");
    animation:grainDrift 7s steps(6) infinite;
}
@keyframes grainDrift {0%{transform:translate(0,0)}100%{transform:translate(-6%,-4%)}}

#MainMenu, footer {visibility:hidden}
[data-testid="stHeader"] {background:transparent}

.stMarkdown, .stMarkdown p {font-family:'Manrope'; color:#d8cfd6}
label, [data-testid="stWidgetLabel"] {color:#9a8f9e !important; font-family:'Spline Sans Mono';
    font-size:11px; letter-spacing:.16em; text-transform:uppercase}

.ds-wordmark {font-family:'Fraunces'; font-weight:900; font-size:58px; line-height:.92;
    letter-spacing:-1.5px; color:#f4ede1; margin:0; font-variation-settings:"opsz" 144}
.ds-wordmark em {font-style:italic; font-weight:500; color:#f0a93b}
.ds-kicker {font-family:'Spline Sans Mono'; font-size:11px; letter-spacing:.34em;
    text-transform:uppercase; color:#8b8090; margin-top:10px}
.ds-label {font-family:'Spline Sans Mono'; font-size:11px; letter-spacing:.3em;
    text-transform:uppercase; color:#f0a93b; margin:26px 0 12px;
    display:flex; align-items:center; gap:10px}
.ds-label::after {content:""; flex:1; height:1px; background:linear-gradient(90deg,rgba(240,169,59,.4),transparent)}
.ds-hint {font-family:'Manrope'; color:#8b8090; font-size:13px; line-height:1.6}

.ds-pill {display:inline-flex; align-items:center; gap:9px; font-family:'Spline Sans Mono';
    font-size:11px; letter-spacing:.14em; padding:8px 15px; border:1px solid rgba(244,237,225,.14);
    border-radius:999px; color:#d8cfd6; background:rgba(244,237,225,.04)}
.ds-dot {width:8px; height:8px; border-radius:50%; background:#5fc6c0; animation:pulse 1.8s infinite}
.ds-dot.slow {background:#f0a93b; animation-name:pulse-a}
@keyframes pulse   {0%{box-shadow:0 0 0 0 rgba(95,198,192,.55)}70%{box-shadow:0 0 0 9px rgba(95,198,192,0)}100%{box-shadow:0 0 0 0 rgba(95,198,192,0)}}
@keyframes pulse-a {0%{box-shadow:0 0 0 0 rgba(240,169,59,.55)}70%{box-shadow:0 0 0 9px rgba(240,169,59,0)}100%{box-shadow:0 0 0 0 rgba(240,169,59,0)}}

.ds-rule {height:2px; margin:22px 0 6px;
    background:linear-gradient(90deg,#f0a93b,#d98c8c,#5fc6c0,#f0a93b);
    background-size:300% 100%; animation:slide 7s linear infinite}
@keyframes slide {to {background-position:300% 0}}

.ds-ticker {overflow:hidden; border-top:1px solid rgba(244,237,225,.08);
    border-bottom:1px solid rgba(244,237,225,.08); margin:14px 0 4px; padding:9px 0;
    -webkit-mask-image:linear-gradient(90deg,transparent,black 6%,black 94%,transparent);
            mask-image:linear-gradient(90deg,transparent,black 6%,black 94%,transparent)}
.ds-track {display:inline-flex; white-space:nowrap; will-change:transform; animation:marquee 26s linear infinite}
.ds-ticker:hover .ds-track {animation-play-state:paused}
.ds-track span {font-family:'Spline Sans Mono'; font-size:11px; letter-spacing:.18em;
    color:#9a8f9e; padding:0 18px; text-transform:uppercase}
.ds-track span b {color:#f0a93b; font-weight:600}
.ds-track .sep {color:#5fc6c0}
@keyframes marquee {to {transform:translateX(-50%)}}

[data-testid="stTextArea"] textarea {
    background:rgba(244,237,225,.03); border:1px solid rgba(244,237,225,.10);
    border-left:3px solid #f0a93b; border-radius:4px 16px 16px 4px; color:#f4ede1;
    font-family:'Fraunces'; font-size:18px; line-height:1.5; font-variation-settings:"opsz" 60;
    transition:border-color .2s, box-shadow .25s, background .2s}
[data-testid="stTextArea"] textarea:focus {
    border-left-color:#d98c8c; background:rgba(244,237,225,.05); outline:none;
    box-shadow:0 0 0 3px rgba(240,169,59,.13), 0 12px 36px -16px rgba(240,169,59,.3)}

.ds-chips {display:flex; flex-wrap:wrap; gap:8px; margin:4px 0 18px}
.ds-chip {font-family:'Spline Sans Mono'; font-size:11px; letter-spacing:.08em; color:#c9bfc8;
    padding:6px 12px; border:1px solid rgba(244,237,225,.12); border-radius:999px;
    background:rgba(244,237,225,.03)}
.ds-chip b {color:#5fc6c0; font-weight:600; margin-right:7px; text-transform:uppercase; font-size:9.5px; letter-spacing:.18em}

.stButton > button, .stDownloadButton > button {
    font-family:'Spline Sans Mono'; font-weight:600; letter-spacing:.12em; text-transform:uppercase;
    font-size:11.5px; color:#d8cfd6; background:rgba(244,237,225,.05);
    border:1px solid rgba(244,237,225,.15); border-radius:6px; transition:transform .15s, box-shadow .2s, border-color .2s}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform:translateY(-2px); border-color:#5fc6c0; color:#fff; box-shadow:0 12px 26px -14px rgba(95,198,192,.5)}
.stButton > button[kind="primary"] {
    font-family:'Fraunces'; font-weight:900; font-size:16px; letter-spacing:.04em; color:#1a1206;
    border:none; padding:15px; background:linear-gradient(135deg,#f6c469,#f0a93b 55%,#d98c8c);
    clip-path:polygon(0 0, calc(100% - 16px) 0, 100% 16px, 100% 100%, 16px 100%, 0 calc(100% - 16px));
    box-shadow:0 14px 36px -12px rgba(240,169,59,.5)}
.stButton > button[kind="primary"]:hover {transform:translateY(-2px); color:#1a1206;
    box-shadow:0 20px 46px -12px rgba(240,169,59,.7)}

[data-testid="stSidebar"] {background:linear-gradient(180deg,#0c0910,#0e0b12)}
[data-baseweb="select"] > div, [data-baseweb="input"] input {
    background:rgba(244,237,225,.045) !important; border-color:rgba(244,237,225,.13) !important; color:#f4ede1 !important}
[data-baseweb="popover"] [role="option"]:hover {background:rgba(240,169,59,.16)}
[data-baseweb="slider"] [role="slider"] {background:#f0a93b !important; border-color:#f0a93b !important}

.stProgress > div > div > div {
    background:linear-gradient(90deg,#f0a93b,#d98c8c,#5fc6c0,#f0a93b) !important;
    background-size:250% 100% !important; animation:slide 2s linear infinite !important; border-radius:99px}

/* photo-developing reveal on the REAL image */
[data-testid="stImage"] {position:relative}
[data-testid="stImage"] img {border-radius:5px 20px 5px 20px; border:1px solid rgba(244,237,225,.12);
    transition:transform .4s, box-shadow .4s; animation:develop 1.1s cubic-bezier(.2,.7,.2,1) both}
[data-testid="stImage"] img:hover {transform:scale(1.015) rotate(-.25deg);
    box-shadow:0 28px 64px -26px rgba(0,0,0,.9)}
[data-testid="stImage"]::after {
    content:""; position:absolute; inset:0; pointer-events:none; border-radius:5px 20px 5px 20px;
    background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='140' height='140'><filter id='m'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/></filter><rect width='100%25' height='100%25' filter='url(%23m)'/></svg>");
    mix-blend-mode:overlay; animation:grainOut 1.1s ease-out both}
@keyframes develop {
    0%   {filter:grayscale(1) sepia(.5) brightness(.5) contrast(.8) blur(3px); opacity:.2}
    45%  {filter:grayscale(.35) sepia(.2) brightness(.85) contrast(.95) blur(1px); opacity:.8}
    100% {filter:none; opacity:1}}
@keyframes grainOut {0%{opacity:.45}100%{opacity:0}}

.ds-edge {position:relative; padding:6px 4px 6px 24px; font-family:'Spline Sans Mono';
    font-size:12px; line-height:2.2; color:#c9bfc8}
.ds-edge::before {content:""; position:absolute; left:2px; top:0; bottom:0; width:9px; opacity:.5;
    background-image:repeating-linear-gradient(180deg,rgba(244,237,225,.7) 0 7px,transparent 7px 20px)}
.ds-edge b {color:#f0a93b; font-weight:600; margin-right:9px; text-transform:uppercase;
    font-size:9.5px; letter-spacing:.2em}

.ds-empty {border:1px dashed rgba(244,237,225,.16); border-radius:6px 22px 6px 22px;
    padding:46px 24px; text-align:center; background:rgba(244,237,225,.02)}
.ds-empty .t {font-family:'Fraunces'; font-style:italic; font-size:22px; color:#c9bfc8}
.ds-empty .s {font-family:'Spline Sans Mono'; font-size:11px; letter-spacing:.2em; color:#8b8090;
    margin-top:10px; text-transform:uppercase}

.ds-strip {background:#0a0709; border:1px solid rgba(244,237,225,.1); border-radius:8px;
    padding:12px 16px; box-shadow:0 24px 60px -30px rgba(0,0,0,.9); overflow:hidden}
.ds-holes {height:9px; margin:7px 0; opacity:.5; border-radius:2px;
    background-image:repeating-linear-gradient(90deg,rgba(244,237,225,.8) 0 9px,transparent 9px 24px)}
.ds-frames {display:flex; gap:14px; overflow-x:auto; padding:6px 2px; scrollbar-width:thin}
.ds-frame {flex:0 0 auto; width:158px}
.ds-frame img {width:100%; height:118px; object-fit:cover; display:block; border-radius:3px;
    border:1px solid rgba(244,237,225,.18); transition:transform .3s, box-shadow .3s}
.ds-frame:hover img {transform:scale(1.05); box-shadow:0 14px 30px -14px rgba(240,169,59,.5)}
.ds-frame .num {font-family:'Spline Sans Mono'; font-size:10px; color:#f0a93b; letter-spacing:.14em; margin-top:7px}
.ds-frame .cap {font-family:'Spline Sans Mono'; font-size:10px; color:#8b8090;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis}

[data-testid="stCaptionContainer"] {font-family:'Spline Sans Mono'; font-size:10.5px; color:#8b8090}
[data-testid="stExpander"] {background:rgba(244,237,225,.03); border:1px solid rgba(244,237,225,.09); border-radius:10px}
[data-testid="stExpander"] summary {font-family:'Spline Sans Mono'; font-size:11px; letter-spacing:.18em; text-transform:uppercase; color:#9a8f9e}
"""
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)

# ──────────────────────────── config (UI knobs only) ──────────────
MODELS = {
    "Tiny SD · lightest (~800MB)": "segmind/tiny-sd",          # BEST FOR STREAMLIT RAM
    "SD Turbo · ultra fast (1-4 steps)": "stabilityai/sd-turbo",
    "SD 1.5 · base (notebook)": "runwayml/stable-diffusion-v1-5",
}
SIZES = {"512 × 512 — square": (512, 512),
         "512 × 768 — portrait": (512, 768),
         "768 × 512 — landscape": (768, 512)}
SURPRISES = [
    "lonely mountain with green grass under a wide sky",
    "a lighthouse made of stained glass standing in a storm, cinematic light",
    "isometric cozy coffee shop inside a giant seashell, warm lanterns, 3d render",
    "a fox spirit wearing a porcelain mask, floating paper lanterns, ukiyo-e",
    "an astronaut fishing on the rings of saturn, retro sci-fi poster",
    "macro photo of a clockwork dragonfly, brass gears, iridescent wings",
]

# ──────────────────────────── helpers (UI only) ───────────────────
@st.cache_resource(show_spinner=False)
def load_pipe(model_id):
    """Loads the model. Optimized dtype & RAM slicing for Streamlit."""
    # float16 on GPU saves half the VRAM. float32 on CPU is required by PyTorch.
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(DEVICE)
    if DEVICE == "cpu":                      # keep peak RAM low so CPU actually finishes
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    return pipe

def build_strip(history):
    frames = []
    for i, item in enumerate(history[:8]):
        thumb = item["image"].copy(); thumb.thumbnail((300, 300))
        buf = io.BytesIO(); thumb.save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        cap = item["prompt"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        frames.append(
            f'<div class="ds-frame" title="{cap}">'
            f'<img src="data:image/png;base64,{b64}" alt="frame">'
            f'<div class="num">◉ {i + 1:02d} · #{item["seed"]}</div>'
            f'<div class="cap">{cap[:34]}</div></div>')
    return ('<div class="ds-strip"><div class="ds-holes"></div>'
            f'<div class="ds-frames">{"".join(frames)}</div><div class="ds-holes"></div></div>')

def do_surprise():
    st.session_state.prompt_box = random.choice(SURPRISES)
def do_clear():
    st.session_state.history = []
def do_restore(seed):
    for item in st.session_state.history:
        if item["seed"] == seed:
            st.session_state.last = item
            st.session_state.prompt_box = item["prompt"]
            break

st.session_state.setdefault("history", [])
st.session_state.setdefault("last", None)
st.session_state.setdefault("prompt_box", "Lonely Mountain with green grass")

# ──────────────────────────── sidebar ─────────────────────────────
with st.sidebar:
    st.markdown('<div class="ds-label" style="margin-top:4px">⚙ Engine room</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ds-hint">Device · <b style="color:#f0a93b">{DEVICE.upper()}</b></div>',
                unsafe_allow_html=True)
    model_name = st.selectbox("Model", list(MODELS) + ["Custom…"])
    model_id = (st.text_input("HuggingFace model ID", "segmind/tiny-sd")
                if model_name == "Custom…" else MODELS[model_name])
    steps    = st.slider("Sampling steps", 1, 80, 15)  # Lower default for light models
    guidance = st.slider("Guidance scale", 0.0, 20.0, 7.5, 0.5)
    size     = st.selectbox("Canvas", list(SIZES))
    seed     = st.number_input("Seed  (−1 = random)", -1, 2**31 - 1, -1)
    st.markdown('<div class="ds-hint">Using <b>Tiny SD</b> or <b>Turbo</b> makes it fast/light.<br>'
                'Same seed + same prompt = same image.</div>', unsafe_allow_html=True)

# ──────────────────────────── header ──────────────────────────────
status = (f"CUDA · {torch.cuda.get_device_name(0)}" if DEVICE == "cuda"
          else "CPU · real generation")
top = st.columns([4, 1.5])
with top[0]:
    st.markdown('<h1 class="ds-wordmark">Atelier <em>//</em> Latent</h1>'
                '<div class="ds-kicker">a darkroom for the latent space · text → image</div>',
                unsafe_allow_html=True)
with top[1]:
    st.markdown(f'<div style="text-align:right;margin-top:18px"><span class="ds-pill">'
                f'<span class="ds-dot{"" if DEVICE == "cuda" else " slow"}"></span>{status}</span></div>',
                unsafe_allow_html=True)
st.markdown('<div class="ds-rule"></div>', unsafe_allow_html=True)

tok = [f'<span>model · <b>{model_id.split("/")[-1]}</b></span>',
       f'<span>steps · <b>{steps}</b></span>', f'<span>cfg · <b>{guidance}</b></span>',
       f'<span>canvas · <b>{size.split(" —")[0].strip()}</b></span>',
       f'<span>device · <b>{DEVICE}</b></span>']
seq = '<span class="sep">◆</span>'.join(tok)
st.markdown(f'<div class="ds-ticker"><div class="ds-track">{seq}<span class="sep">◆</span>{seq}</div></div>',
            unsafe_allow_html=True)

# ──────────────────────── prompt console ──────────────────────────
st.markdown('<div class="ds-label">✦ Composition sheet</div>', unsafe_allow_html=True)
prompt = st.text_area("Describe the image you want to exist", height=120, key="prompt_box",
                      placeholder="a lighthouse made of stained glass, storm light, ultra detailed …")
with st.expander("Negative prompt — what to keep out"):
    negative = st.text_input("Negative", "blurry, low quality, watermark, extra fingers",
                             label_visibility="collapsed")

st.markdown(f'<div class="ds-chips"><span class="ds-chip"><b>steps</b>{steps}</span>'
            f'<span class="ds-chip"><b>cfg</b>{guidance}</span>'
            f'<span class="ds-chip"><b>seed</b>{"rand" if seed == -1 else seed}</span></div>',
            unsafe_allow_html=True)

if DEVICE == "cpu":
    st.info("No GPU detected — a **real** image will be generated on CPU. "
            "Using **Tiny SD** or **Turbo** keeps it fast and prevents memory crashes.")

b1, b2, b3 = st.columns([3, 1.4, 1.4])
generate = b1.button("◉ Generate", type="primary", use_container_width=True)
b2.button("✦ Surprise me", on_click=do_surprise, use_container_width=True)
b3.button("✕ Clear strip", on_click=do_clear, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
#  THE CORE — your notebook, unchanged in spirit (Steps 1 → 4)
# ══════════════════════════════════════════════════════════════════
if generate:
    if not prompt.strip():
        st.warning("Give the model something to dream about first ✍️")
        st.stop()

    w, h = SIZES[size]
    used_seed = random.randint(0, 2**31 - 1) if seed == -1 else seed
    t0 = time.perf_counter()

    # Smart overrides for Turbo models (they require specific settings to not break)
    is_turbo = "turbo" in model_id.lower() or "lightning" in model_id.lower()
    if is_turbo:
        w, h = 512, 512       # SD Turbo strictly requires 512x512
        steps = min(steps, 4) # Turbo only needs 1-4 steps
        guidance = 0.0        # Turbo requires NO guidance scale

    bar = st.progress(0.0, text="loading model…")

    # Step 1: load the model (cached after first run)
    pipe = load_pipe(model_id)

    # live progress while the generation runs
    def on_step(p, step_index, timestep, kwargs):
        bar.progress((step_index + 1) / steps, text=f"generating · step {step_index + 1}/{steps}")
        return kwargs

    # Step 3: generate the image  ->  pipe(prompt).images[0]
    with torch.inference_mode():
        try:
            image = pipe(prompt, negative_prompt=negative,
                         num_inference_steps=steps, guidance_scale=guidance,
                         width=w, height=h,
                         generator=torch.Generator(DEVICE).manual_seed(used_seed),
                         callback_on_step_end=on_step).images[0]
        except TypeError:   # older diffusers without the callback kwarg
            image = pipe(prompt, negative_prompt=negative,
                         num_inference_steps=steps, guidance_scale=guidance,
                         width=w, height=h,
                         generator=torch.Generator(DEVICE).manual_seed(used_seed)).images[0]

    # Step 4: save the image (mirrors your notebook; never breaks the UI)
    try:
        image.save("Generated_Image.png")
    except Exception:
        pass

    elapsed = time.perf_counter() - t0
    bar.progress(1.0, text=f"done · {elapsed:.1f}s ✦")

    st.session_state.last = dict(image=image, prompt=prompt, seed=used_seed, steps=steps,
                                 guidance=guidance, size=f"{w}×{h}", model=model_id, time=elapsed)
    st.session_state.history.insert(0, st.session_state.last)
# ══════════════════════════════════════════════════════════════════

# ───────────────────────── latest render ──────────────────────────
st.markdown('<div class="ds-label">✦ Print</div>', unsafe_allow_html=True)
if st.session_state.last:
    r = st.session_state.last
    view, rail = st.columns([3, 1.1])
    with view:
        st.image(r["image"], use_container_width=True)
    with rail:
        buf = io.BytesIO(); r["image"].save(buf, "PNG")
        st.download_button("⬇ Save PNG", buf.getvalue(), f"latent-{r['seed']}.png",
                           "image/png", use_container_width=True)
        st.markdown(f'<div class="ds-edge"><b>seed</b>{r["seed"]}<br><b>steps</b>{r["steps"]}<br>'
                    f'<b>cfg</b>{r["guidance"]}<br><b>size</b>{r["size"]}<br>'
                    f'<b>time</b>{r["time"]:.1f}s<br>'
                    f'<b>model</b>{r["model"].split("/")[-1]}</div>', unsafe_allow_html=True)
        st.caption(f"“{r['prompt']}”")
else:
    st.markdown('<div class="ds-empty"><div class="t">the tray is empty</div>'
                '<div class="s">write a prompt · hit generate · watch it surface</div></div>',
                unsafe_allow_html=True)

# ──────────────────────── film-strip gallery ──────────────────────
if st.session_state.history:
    st.markdown('<div class="ds-label">✦ Contact sheet</div>', unsafe_allow_html=True)
    st.markdown(build_strip(st.session_state.history), unsafe_allow_html=True)
    rc = st.columns(min(len(st.session_state.history), 8))
    for i, item in enumerate(st.session_state.history[:8]):
        with rc[i]:
            st.button(f"↺ #{item['seed']}", key=f"r{item['seed']}",
                      on_click=do_restore, args=(item["seed"],), use_container_width=True)

st.markdown('<div class="ds-kicker" style="margin-top:34px">atelier // latent · streamlit + 🤗 diffusers</div>',
            unsafe_allow_html=True)
