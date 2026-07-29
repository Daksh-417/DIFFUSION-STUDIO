"""
DIFFUSION // STUDIO — a pretty face for Stable Diffusion
Run:  streamlit run app.py
Install:  pip install streamlit torch diffusers transformers accelerate safetensors pillow numpy
"""

import io
import random
import time

import numpy as np
import streamlit as st
import torch
from PIL import Image

# ────────────────────────────── page ──────────────────────────────
st.set_page_config(page_title="Diffusion Studio", page_icon="✦", layout="wide")

# ────────────────────────────── theme ─────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;700;800&family=Instrument+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

/* ambient layered canvas */
.stApp {
    color:#e8efee;
    background:
        radial-gradient(1100px 620px at 88% -12%, rgba(255,158,66,.16), transparent 62%),
        radial-gradient(900px 700px at -8% 108%, rgba(45,212,191,.12), transparent 58%),
        radial-gradient(760px 520px at 55% 118%, rgba(255,107,94,.08), transparent 60%),
        #0c1214;
}
.stApp::before {  /* faint blueprint grid */
    content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
    background-image:
        linear-gradient(rgba(255,255,255,.028) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.028) 1px, transparent 1px);
    background-size:44px 44px;
    -webkit-mask-image:radial-gradient(ellipse 90% 70% at 50% 18%, black, transparent 78%);
            mask-image:radial-gradient(ellipse 90% 70% at 50% 18%, black, transparent 78%);
}
#MainMenu, footer {visibility:hidden}
[data-testid="stHeader"] {background:transparent}

.stMarkdown, .stMarkdown p {font-family:'Instrument Sans'; color:#dbe6e4}
label, [data-testid="stWidgetLabel"] {color:#93a5a3 !important}

/* type helpers */
.ds-logo {font-family:'Syne'; font-weight:800; font-size:46px; line-height:1; letter-spacing:-.5px; color:#f4f1ea; margin:0}
.ds-logo span {color:#ff9e42}
.ds-tag {font-family:'JetBrains Mono'; font-size:11px; letter-spacing:.22em; text-transform:uppercase; color:#7d8b8d; margin-top:8px}
.ds-label {font-family:'JetBrains Mono'; font-size:11px; letter-spacing:.26em; text-transform:uppercase; color:#ff9e42; margin:22px 0 10px}
.ds-hint {font-family:'Instrument Sans'; color:#8fa0a2; font-size:13px; line-height:1.6}

/* live status pill */
.ds-pill {display:inline-flex; align-items:center; gap:9px; font-family:'JetBrains Mono'; font-size:11px;
    letter-spacing:.14em; padding:8px 15px; border:1px solid rgba(255,255,255,.13); border-radius:999px;
    color:#cfe3e0; background:rgba(255,255,255,.045)}
.ds-dot {width:8px; height:8px; border-radius:50%; background:#3ddc97; animation:pulse 1.8s infinite}
.ds-dot.amber {background:#ffb454; animation-name:pulse-a}
@keyframes pulse   {0%{box-shadow:0 0 0 0 rgba(61,220,151,.55)} 70%{box-shadow:0 0 0 9px rgba(61,220,151,0)} 100%{box-shadow:0 0 0 0 rgba(61,220,151,0)}}
@keyframes pulse-a {0%{box-shadow:0 0 0 0 rgba(255,180,84,.55)} 70%{box-shadow:0 0 0 9px rgba(255,180,84,0)} 100%{box-shadow:0 0 0 0 rgba(255,180,84,0)}}

/* shimmer divider */
.ds-rule {height:2px; margin:24px 0 8px; opacity:.75;
    background:linear-gradient(90deg,#ff9e42,#ff6b5e,#2dd4bf,#ff9e42);
    background-size:300% 100%; animation:slide 7s linear infinite}
@keyframes slide {to {background-position:300% 0}}

/* prompt console */
[data-testid="stTextArea"] textarea {
    background:rgba(255,255,255,.035); border:1px solid rgba(255,255,255,.10); border-left:3px solid #ff9e42;
    border-radius:4px 16px 16px 4px; color:#eef3f2; font-family:'Instrument Sans'; font-size:16.5px; line-height:1.55;
    transition:border-color .2s, box-shadow .25s, background .2s}
[data-testid="stTextArea"] textarea:focus {
    border-left-color:#ff6b5e; background:rgba(255,255,255,.05); outline:none;
    box-shadow:0 0 0 3px rgba(255,158,66,.14), 0 10px 34px -14px rgba(255,158,66,.3)}

/* buttons */
.stButton > button, .stDownloadButton > button {
    font-family:'JetBrains Mono'; font-weight:600; letter-spacing:.12em; text-transform:uppercase; font-size:11.5px;
    color:#dfe9e7; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.15); border-radius:6px;
    transition:transform .15s, box-shadow .2s, border-color .2s}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform:translateY(-2px); border-color:#2dd4bf; color:#fff;
    box-shadow:0 12px 26px -14px rgba(45,212,191,.55)}
.stButton > button[kind="primary"] {
    font-family:'Syne'; font-weight:800; font-size:15px; letter-spacing:.16em; color:#1a1206; border:none; padding:15px;
    background:linear-gradient(135deg,#ffb454,#ff9e42 55%,#ff6b5e);
    clip-path:polygon(0 0, calc(100% - 16px) 0, 100% 16px, 100% 100%, 16px 100%, 0 calc(100% - 16px));
    box-shadow:0 14px 36px -12px rgba(255,140,60,.55)}
.stButton > button[kind="primary"]:hover {
    transform:translateY(-2px); color:#1a1206; box-shadow:0 20px 46px -12px rgba(255,140,60,.75)}

/* sidebar */
[data-testid="stSidebar"] {background:linear-gradient(180deg,#0a1012,#0b1517)}
[data-baseweb="select"] > div, [data-baseweb="input"] input {
    background:rgba(255,255,255,.045) !important; border-color:rgba(255,255,255,.13) !important; color:#eef3f2 !important}
[data-baseweb="popover"] [role="option"]:hover {background:rgba(255,158,66,.16)}
[data-baseweb="slider"] [role="slider"] {background:#ff9e42 !important; border-color:#ff9e42 !important}

/* progress — animated stripes */
.stProgress > div > div > div {
    background:linear-gradient(90deg,#ff9e42,#ff6b5e,#2dd4bf,#ff9e42) !important;
    background-size:250% 100% !important; animation:slide 2s linear infinite !important; border-radius:99px}

/* output + gallery */
[data-testid="stImage"] img {border-radius:6px 22px 6px 22px; border:1px solid rgba(255,255,255,.11);
    transition:transform .35s, box-shadow .35s}
[data-testid="stImage"] img:hover {transform:scale(1.015); box-shadow:0 26px 60px -24px rgba(0,0,0,.85)}
[data-testid="stCaptionContainer"] {font-family:'JetBrains Mono'; font-size:10.5px; color:#7d8b8d}
[data-testid="stExpander"] {background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.09); border-radius:10px}
[data-testid="stExpander"] summary {font-family:'JetBrains Mono'; font-size:11px; letter-spacing:.18em; text-transform:uppercase; color:#9fb4b1}

.ds-meta {font-family:'JetBrains Mono'; font-size:12px; line-height:2.1; color:#c6d3d1;
    border:1px dashed rgba(255,255,255,.15); border-radius:4px 14px 4px 14px;
    padding:12px 16px; background:rgba(255,255,255,.03)}
.ds-meta b {color:#ff9e42; font-weight:600; margin-right:8px; text-transform:uppercase; font-size:9.5px; letter-spacing:.2em}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# ──────────────────────────── config ──────────────────────────────
MODELS = {
    "SD 1.5 · base": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "DreamShaper 8 · artistic": "Lykon/dreamshaper-8",
    "SD 2.1 · base": "stabilityai/stable-diffusion-2-1",
}
SIZES = {
    "512 × 512 — square": (512, 512),
    "512 × 768 — portrait": (512, 768),
    "768 × 512 — landscape": (768, 512),
}
SCHEDULERS = {
    "Euler a": "EulerAncestralDiscreteScheduler",
    "DPM++ 2M": "DPMSolverMultistepScheduler",
    "DDIM": "DDIMScheduler",
}
SURPRISES = [
    "a lighthouse made of stained glass standing in a storm, cinematic light, ultra detailed",
    "isometric cozy coffee shop inside a giant seashell, warm lanterns, pastel fog, 3d render",
    "a fox spirit wearing a porcelain mask, floating paper lanterns, ukiyo-e style, gold leaf",
    "brutalist treehouse city at golden hour, hanging gardens, volumetric light, matte painting",
    "an astronaut fishing on the rings of saturn, retro sci-fi poster, grainy film texture",
    "macro photo of a clockwork dragonfly, brass gears, iridescent wings, studio lighting",
    "underwater train station with whales passing by, art deco tiles, god rays, dreamy",
    "a library carved inside a redwood trunk, spiral stairs, dust motes in sunbeams, storybook art",
]

# ──────────────────────────── helpers ─────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipe(model_id):
    """Download + cache the model. Slow once, instant afterwards."""
    import diffusers
    pipe = diffusers.StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    return pipe.to("cuda")


def make_demo_image(w, h):
    """Fake 'diffused' art so the UI is fully testable without a GPU."""
    rng = np.random.default_rng()
    a, b = rng.integers(30, 230, 3), rng.integers(30, 230, 3)
    t = np.linspace(0, 1, w)[None, :, None]
    img = np.repeat((1 - t) * a + t * b, h, axis=0)     # smooth gradient
    img += rng.normal(0, 14, img.shape)                 # film grain
    return Image.fromarray(np.clip(img, 0, 255).astype("uint8"))


def do_surprise():
    st.session_state.prompt_box = random.choice(SURPRISES)

def do_clear():
    st.session_state.history = []

def do_restore(item):
    st.session_state.last = item


st.session_state.setdefault("history", [])
st.session_state.setdefault("last", None)

# ──────────────────────────── sidebar ─────────────────────────────
with st.sidebar:
    st.markdown('<div class="ds-label" style="margin-top:6px">⚙ Engine room</div>', unsafe_allow_html=True)
    demo = st.toggle("Demo mode — no GPU needed", value=not torch.cuda.is_available(),
                     help="Renders placeholder art so you can play with the UI instantly.")
    model_name = st.selectbox("Model", list(MODELS) + ["Custom…"])
    model_id = (st.text_input("HuggingFace model ID", "Lykon/dreamshaper-8")
                if model_name == "Custom…" else MODELS[model_name])
    steps      = st.slider("Sampling steps", 10, 60, 30)
    guidance   = st.slider("Guidance scale", 1.0, 20.0, 7.5, 0.5)
    scheduler  = st.selectbox("Sampler", list(SCHEDULERS))
    size       = st.selectbox("Canvas", list(SIZES))
    seed       = st.number_input("Seed  (−1 = random)", -1, 2**31 - 1, -1)
    st.markdown('<div class="ds-hint">Tip: fewer steps = faster drafts.<br>'
                'Same seed + same prompt = same image.</div>', unsafe_allow_html=True)

# ──────────────────────────── header ──────────────────────────────
gpu_ok = torch.cuda.is_available()
live   = gpu_ok and not demo
status = f"CUDA · {torch.cuda.get_device_name(0)}" if live else "DEMO · no model loaded"

top = st.columns([4, 1.6])
with top[0]:
    st.markdown('<h1 class="ds-logo">DIFFUSION<span>//</span>STUDIO</h1>'
                '<div class="ds-tag">text → image · stable diffusion playground</div>', unsafe_allow_html=True)
with top[1]:
    st.markdown(f'<div style="text-align:right; margin-top:14px"><span class="ds-pill">'
                f'<span class="ds-dot{"" if live else " amber"}"></span>{status}</span></div>',
                unsafe_allow_html=True)
st.markdown('<div class="ds-rule"></div>', unsafe_allow_html=True)

# ──────────────────────── prompt console ──────────────────────────
st.markdown('<div class="ds-label">✦ Prompt console</div>', unsafe_allow_html=True)
prompt = st.text_area("Describe the image you want to exist", height=110, key="prompt_box",
                      placeholder="a lighthouse made of stained glass, storm light, ultra detailed, 8k …")
with st.expander("Negative prompt — what to keep out"):
    negative = st.text_input("Negative", "blurry, low quality, watermark, extra fingers",
                             label_visibility="collapsed")

b1, b2, b3 = st.columns([3, 1.5, 1.5])
generate = b1.button("⚡ Generate", type="primary", use_container_width=True)
b2.button("🎲 Surprise me", on_click=do_surprise, use_container_width=True)
b3.button("✕ Clear history", on_click=do_clear, use_container_width=True)

# ─────────────────────────── generate ─────────────────────────────
if generate:
    if not prompt.strip():
        st.warning("Give the model something to dream about first ✍️")
        st.stop()
    if not demo and not gpu_ok:
        st.error("No CUDA GPU found — flip on **Demo mode** in the sidebar, or run on a GPU machine.")
        st.stop()

    w, h = SIZES[size]
    used_seed = random.randint(0, 2**31 - 1) if seed == -1 else seed
    bar = st.progress(0.0, text="warming up the latent space…")

    if demo:
        for s in range(steps):                       # pretend to diffuse
            time.sleep(0.03)
            bar.progress((s + 1) / steps, text=f"diffusing · step {s + 1}/{steps}")
        image = make_demo_image(w, h)
    else:
        import diffusers
        with st.spinner("Loading model… (first run downloads a few GB)"):
            pipe = load_pipe(model_id)
        pipe.scheduler = getattr(diffusers, SCHEDULERS[scheduler]).from_config(pipe.scheduler.config)

        def on_step(p, step_index, timestep, kwargs):  # live progress bar
            bar.progress((step_index + 1) / steps, text=f"diffusing · step {step_index + 1}/{steps}")
            return kwargs

        with torch.inference_mode():
            image = pipe(prompt, negative_prompt=negative,
                         num_inference_steps=steps, guidance_scale=guidance,
                         width=w, height=h,
                         generator=torch.Generator("cuda").manual_seed(used_seed),
                         callback_on_step_end=on_step).images[0]

    bar.progress(1.0, text="done ✦")
    st.session_state.last = dict(image=image, prompt=prompt, seed=used_seed,
                                 steps=steps, guidance=guidance, size=f"{w}×{h}", model=model_id)
    st.session_state.history.insert(0, st.session_state.last)

# ───────────────────────── latest render ──────────────────────────
if st.session_state.last:
    r = st.session_state.last
    st.markdown('<div class="ds-label">✦ Latest render</div>', unsafe_allow_html=True)
    view, meta = st.columns([3, 1.2])
    with view:
        st.image(r["image"], use_container_width=True)
    with meta:
        buf = io.BytesIO(); r["image"].save(buf, "png")
        st.download_button("⬇ Save PNG", buf.getvalue(), f"diffusion-{r['seed']}.png",
                           "image/png", use_container_width=True)
        st.markdown(f'<div class="ds-meta"><b>seed</b>{r["seed"]}<br><b>steps</b>{r["steps"]}<br>'
                    f'<b>cfg</b>{r["guidance"]}<br><b>size</b>{r["size"]}<br>'
                    f'<b>model</b>{r["model"]}</div>', unsafe_allow_html=True)
        st.caption(f"“{r['prompt']}”")

# ──────────────────────── session gallery ─────────────────────────
if st.session_state.history:
    st.markdown('<div class="ds-label">✦ Session gallery</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, item in enumerate(st.session_state.history[:12]):
        with cols[i % 4]:
            st.image(item["image"], use_container_width=True)
            st.caption(f"#{item['seed']} · {item['prompt'][:36]}…")
            st.button("↺ restore", key=f"r{i}", on_click=do_restore, args=(item,),
                      use_container_width=True)

st.markdown('<div class="ds-tag" style="margin-top:36px">diffusion//studio · streamlit + 🤗 diffusers</div>',
            unsafe_allow_html=True)
