# app.py
import os
import re
import time
import json
import textwrap
from datetime import datetime, timedelta
import streamlit as st
import firebase_admin
from firebase_admin import firestore, credentials

# ===========================
# CONFIG
# ===========================

MODEL_NAME = "gemini-1.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MAX_CHARS_FREE = 10000
MAX_CHARS_PRO = 50000
GEMINI_HARD_LIMIT_CHARS = 30000

# Firebase
FIREBASE_CRED_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT")
if FIREBASE_CRED_JSON:
    cred = credentials.Certificate(json.loads(FIREBASE_CRED_JSON))
    firebase_admin.initialize_app(cred)
FIRESTORE_DB = firestore.client() if firebase_admin.apps else None

# Gemini
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    HAS_GENAI = True
except:
    HAS_GENAI = False

# ===========================
# AI DETECTION PATTERNS (Reddit + Dawn leaks)
# ===========================

AI_PROMPT_PATTERNS = [
    r"\bas an a[i1l] language model\b",
    r"\bi am an a[i1l]\b",
    r"\bi'?m an a[i1l]\b",
    r"\blanguage model\b",
    r"\[system prompt[:\]]",
    r"\bi'?d be happy to assist\b",
    r"\bi'?m here to help\b",
    r"\bif you want, i can also\b",
    r"\bdo you want me to\b",
]

AI_STYLE_KEYWORDS = [
    "in the world of", "in today’s digital age", "it is crucial to",
    "let’s delve into", "dive deep", "not only", "but also", "play a crucial role"
]

FORMAL_GREETINGS = ["i hope this email finds you well"]

# ===========================
# UTILS
# ===========================

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def analyze_style(text):
    risks = []
    t, tl = text, text.lower()

    # Em dashes
    if t.count("—") >= 4:
        risks.append({"snippet": "Multiple em dashes (—).", "reason": "AI overuse.", "fix": "Use commas or periods."})

    # Semicolons
    if t.count(";") >= 3:
        risks.append({"snippet": "Multiple semicolons.", "reason": "Rare in journalism.", "fix": "Break into shorter sentences."})

    # Formal greetings
    for g in FORMAL_GREETINGS:
        if g in tl:
            risks.append({"snippet": g, "reason": "Email-style AI greeting.", "fix": "Remove or rephrase."})

    # Repetitive keywords
    for kw in ["crucial", "delve", "dive deep"]:
        if tl.count(kw) >= 2:
            risks.append({"snippet": f"'{kw}' ×{tl.count(kw)}", "reason": "AI filler word.", "fix": "Vary vocabulary."})

    # Formulaic intros
    for intro in AI_STYLE_KEYWORDS:
        if intro in tl:
            risks.append({"snippet": intro, "reason": "AI explainer cliché.", "fix": "Make opening specific."})

    # No contractions
    words = re.findall(r"\b\w+\b", tl)
    contractions = sum(1 for w in words if "'" in w)
    if len(words) > 80 and contractions <= 1:
        risks.append({"snippet": "Few contractions.", "reason": "Stiff AI tone.", "fix": "Add natural contractions."})

    # Uniform sentence length
    sents = split_sentences(t)
    if len(sents) >= 5:
        lens = [len(s.split()) for s in sents]
        if max(lens) - min(lens) <= 8 and sum(lens)/len(lens) > 22:
            risks.append({"snippet": "Uniform sentence length.", "reason": "AI rhythm.", "fix": "Vary sentence length."})

    return risks

def local_qc(text):
    lines = text.splitlines()
    leaks = []
    cleaned_lines = []

    for line in lines:
        s = line.strip()
        if not s:
            cleaned_lines.append(line)
            continue
        if any(re.search(p, s.lower()) for p in AI_PROMPT_PATTERNS):
            leaks.append({"snippet": s, "reason": "AI prompt leak.", "fix": "Remove this line."})
        else:
            cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    style_risks = analyze_style(text)

    prob = 97 if leaks else 80 if style_risks else 40
    sev = 8 if leaks else 5 if style_risks else 1

    return {
        "prompt_leaks": leaks,
        "other_risks": style_risks,
        "ai_probability_score": prob,
        "severity_score": sev,
        "clean_text": cleaned,
        "_meta": {"source": "local", "elapsed": 0.1}
    }

# ===========================
# GEMINI CLEAN
# ===========================

def gemini_clean(text, lang, region, max_chars):
    if not HAS_GENAI or not GEMINI_API_KEY:
        return {"ok": False, "error": "Gemini not available."}

    model = genai.GenerativeModel(MODEL_NAME)
    limit = min(max_chars, GEMINI_HARD_LIMIT_CHARS)
    trimmed = len(text) > limit
    text = text[:limit]

    prompt = f"""You are a senior newspaper editor.

Remove ALL AI meta, chatty tone, system prompts, and formulaic phrases.
Keep facts and meaning 100% intact.
Write like a real journalist for {region}.

Language: {lang}

ARTICLE_START
{text}
ARTICLE_END

Return ONLY the cleaned article. No JSON. No explanation."""

    try:
        start = time.time()
        resp = model.generate_content(prompt, generation_config={"temperature": 0.3, "max_output_tokens": 2048})
        elapsed = time.time() - start
        parts = getattr(resp.candidates[0].content, "parts", []) if resp.candidates else []
        out = "".join(p.text for p in parts if hasattr(p, "text")).strip()
        return {"ok": True, "clean_text": out, "elapsed": elapsed, "trimmed": trimmed}
    except Exception as e:
        return {"ok": False, "error": str(e), "elapsed": 0}

# ===========================
# FIREBASE: CHECK PRO ACCESS
# ===========================

def get_user_plan(email_or_token):
    if not FIRESTORE_DB:
        return {"is_pro": False, "expires": None}

    # Try token first
    doc = FIRESTORE_DB.collection("pro_access").document(email_or_token).get()
    if doc.exists:
        data = doc.to_dict()
        if data.get("expires", 0) > time.time():
            return {"is_pro": True, "expires": data["expires"], "email": data.get("email")}
    
    # Try email
    docs = FIRESTORE_DB.collection("pro_access").where("email", "==", email_or_token).stream()
    for d in docs:
        data = d.to_dict()
        if data.get("expires", 0) > time.time():
            return {"is_pro": True, "expires": data["expires"], "email": data["email"]}
    
    return {"is_pro": False, "expires": None}

# ===========================
# UI
# ===========================

st.set_page_config(page_title="Editorial Guard", layout="centered", page_icon="shield")

# Custom CSS
st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton>button { background: #1E88E5; color: white; font-weight: bold; }
    .metric-card { background: #f8f9fa; padding: 1rem; border-radius: 10px; }
    .pro-badge { background: #FFD700; color: #000; padding: 0.2rem 0.5rem; border-radius: 5px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("Editorial Guard")
st.markdown("**Detect & remove AI leaks before publication**")

# --- Auth via URL ---
qp = st.query_params
token = qp.get("token", [None])[0]
email = qp.get("email", [None])[0]

plan = get_user_plan(token or email or "")
is_pro = plan["is_pro"]
max_chars = MAX_CHARS_PRO if is_pro else MAX_CHARS_FREE
plan_name = "Pro" if is_pro else "Free"

if is_pro:
    expires = datetime.fromtimestamp(plan["expires"]).strftime("%b %d, %Y")
    st.success(f"Pro Active · Expires: {expires}")
else:
    st.info(f"Free Plan · Max {MAX_CHARS_FREE:,} chars · [Upgrade →](https://www.mindscopeai.net/upgrade)")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    st.write(f"**Plan:** {plan_name}")
    st.write(f"**Limit:** {max_chars:,} chars")
    
    language = st.selectbox("Language", ["English", "Urdu", "Hindi", "Spanish", "Other"])
    region = st.selectbox("Region", ["Pakistan", "US", "UK", "India", "Global"])

# --- Input ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Article")
    default = textwrap.dedent("""\
    As an AI language model, I strive to provide helpful responses.
    [SYSTEM PROMPT: Be neutral]
    Inflation in Pakistan has risen sharply due to global pressures.
    """)
    text = st.text_area("", value=default, height=300, label_visibility="collapsed")

with col2:
    st.subheader("Cleaned Output")
    if st.button("Run QC", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Paste text first.")
        else:
            with st.spinner("Analyzing..."):
                data = local_qc(text)
                meta = data["_meta"]

                if len(text) > max_chars:
                    st.error(f"Document too long for {plan_name} plan.")
                elif HAS_GENAI and GEMINI_API_KEY and is_pro:
                    g = gemini_clean(data["clean_text"], language, region, max_chars)
                    if g["ok"]:
                        data["clean_text"] = g["clean_text"]
                        meta.update({"source": "gemini", "gemini_time": g["elapsed"]})

                # --- Results ---
                c1, c2, c3 = st.columns(3)
                c1.metric("AI Risk", f"{data['ai_probability_score']}%")
                c2.metric("Severity", data['severity_score'])
                c3.metric("Plan", plan_name)

                if data["prompt_leaks"]:
                    st.error("Prompt Leaks Found")
                    for l in data["prompt_leaks"]:
                        st.markdown(f"**Remove:** `{l['snippet']}`")

                if data["other_risks"]:
                    st.warning("AI Style Flags")
                    for r in data["other_risks"][:3]:
                        st.markdown(f"• {r['reason']}")

                st.success("Cleaned Text Ready")
                st.code(data["clean_text"], language="text")