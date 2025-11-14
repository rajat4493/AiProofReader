# app.py — CleanCopy: Fixed Firebase Init
import os
import re
import time
import json
import textwrap
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components

# ===========================
# CONFIG
# ===========================
APP_NAME = "CleanCopy"
MODEL_NAME = "gemini-1.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MAX_CHARS_FREE = 10000
MAX_CHARS_PRO = 50000
GEMINI_HARD_LIMIT_CHARS = 30000

# Firebase (Safe Init)
FIRESTORE_DB = None
try:
    import firebase_admin
    from firebase_admin import firestore, credentials

    FIREBASE_CRED_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if FIREBASE_CRED_JSON:
        cred = credentials.Certificate(json.loads(FIREBASE_CRED_JSON))
        # Initialize safely (no duplicates)
        if not firebase_admin._apps:  # Use internal check to avoid error
            firebase_admin.initialize_app(cred)
        FIRESTORE_DB = firestore.client()
    HAS_FIREBASE = FIRESTORE_DB is not None
except ImportError:
    HAS_FIREBASE = False
except Exception as e:
    st.error(f"Firebase setup error: {e}")
    HAS_FIREBASE = False

# Gemini
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    HAS_GENAI = True
except:
    HAS_GENAI = False

# ===========================
# AI DETECTION
# ===========================
AI_PROMPT_PATTERNS = [
    r"\bas an a[i1l] language model\b", r"\bi am an a[i1l]\b", r"\bi'?m an a[i1l]\b",
    r"\blanguage model\b", r"$$ system prompt[: $$]", r"\bi'?d be happy to assist\b",
    r"\bi'?m here to help\b", r"\bif you want, i can also\b", r"\bdo you want me to\b",
    r"\bhere'?s your\b",
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

    if t.count("—") >= 4:
        risks.append({"snippet": "Multiple em dashes (—)", "reason": "AI overuse", "fix": "Use commas"})

    if t.count(";") >= 3:
        risks.append({"snippet": "Multiple semicolons", "reason": "Rare in journalism", "fix": "Break sentences"})

    for g in FORMAL_GREETINGS:
        if g in tl:
            risks.append({"snippet": g, "reason": "AI email tone", "fix": "Remove"})

    for kw in ["crucial", "delve", "dive deep"]:
        count = tl.count(kw)
        if count >= 2:
            risks.append({"snippet": f"'{kw}' ×{count}", "reason": "AI filler", "fix": "Vary words"})

    for intro in AI_STYLE_KEYWORDS:
        if intro in tl:
            risks.append({"snippet": intro, "reason": "AI cliché", "fix": "Be specific"})

    words = re.findall(r"\b\w+\b", tl)
    contractions = sum(1 for w in words if "'" in w)
    if len(words) > 80 and contractions <= 1:
        risks.append({"snippet": "No contractions", "reason": "Stiff AI tone", "fix": "Add 'don't', 'it's'"})

    sents = split_sentences(t)
    if len(sents) >= 5:
        lens = [len(s.split()) for s in sents]
        if lens and max(lens) - min(lens) <= 8 and sum(lens)/len(lens) > 22:
            risks.append({"snippet": "Uniform sentences", "reason": "AI rhythm", "fix": "Vary length"})

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
            leaks.append({"snippet": s, "reason": "AI prompt leak", "fix": "Remove"})
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
        "_meta": {"source": "local"}
    }

# ===========================
# GEMINI CLEAN
# ===========================
def gemini_clean(text, lang, region, max_chars):
    if not HAS_GENAI or not GEMINI_API_KEY:
        return {"ok": False, "error": "Gemini not available"}

    model = genai.GenerativeModel(MODEL_NAME)
    limit = min(max_chars, GEMINI_HARD_LIMIT_CHARS)
    trimmed = len(text) > limit
    text = text[:limit]

    prompt = f"""You are a senior global news editor.

Remove ALL AI meta, chatty tone, system prompts, and formulaic phrases.
Keep facts 100% intact. Write like a real journalist for {region}.
Support {lang}.

ARTICLE_START
{text}
ARTICLE_END

Return ONLY the cleaned article. No JSON."""

    try:
        start = time.time()
        resp = model.generate_content(prompt, generation_config={"temperature": 0.3})
        elapsed = time.time() - start
        out = "".join(getattr(p, "text", "") for p in resp.candidates[0].content.parts).strip()
        return {"ok": True, "clean_text": out or text, "elapsed": elapsed, "trimmed": trimmed}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ===========================
# FIREBASE AUTH (Safe)
# ===========================
def get_user_plan(email_or_token):
    if not HAS_FIREBASE or not FIRESTORE_DB:
        return {"is_pro": False, "error": "Firebase not connected"}

    token = str(email_or_token).strip()
    
    # BLOCK EMPTY / INVALID
    if not token or token == "None":
        return {"is_pro": False}

    try:
        # 1. Try as TOKEN (document ID)
        doc_ref = FIRESTORE_DB.collection("pro_access").document(token)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            if data.get("expires", 0) > time.time():
                return {"is_pro": True, "expires": data["expires"], "email": data.get("email")}

        # 2. Try as EMAIL (search field)
        if "@" in token:  # rough email check
            query = FIRESTORE_DB.collection("pro_access").where("email", "==", token).limit(1).stream()
            for d in query:
                data = d.to_dict()
                if data.get("expires", 0) > time.time():
                    return {"is_pro": True, "expires": data["expires"], "email": data["email"], "token": d.id}

    except Exception as e:
        st.error(f"Pro check failed: {e}")
        return {"is_pro": False, "error": str(e)}
    
    return {"is_pro": False}

# ===========================
# UI
# ===========================
st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="🧹")

st.markdown("""
<style>
    .main { padding: 2rem; max-width: 1200px; margin: auto; }
    .stButton > button { background: #1976D2; color: white; font-weight: bold; }
    .metric { background: #f5f5f5; padding: 1rem; border-radius: 10px; text-align: center; }
    .pro { background: #FFD700; color: #000; padding: 0.3rem 0.6rem; border-radius: 20px; font-weight: bold; }
    h1 { color: #1976D2; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title(f"🧹 {APP_NAME}")
st.markdown("*AI wrote it. We fix it. For journalists worldwide.*")

# Auth
# === AUTH: Token from URL, Email from input ===
qp = st.query_params
token = qp.get("token", [None])[0]  # From URL: ?token=abc123
email_input = st.text_input(
    "Email (optional for Pro)",
    type="default",
    placeholder="you@example.com",
    help="Enter to check Pro status"
)
email = email_input.strip() or ""   # Clean input
token = token or ""                 # Clean token

# === CHECK PLAN (token first, then email) ===
plan = get_user_plan(token or email)
is_pro = plan.get("is_pro", False)
max_chars = MAX_CHARS_PRO if is_pro else MAX_CHARS_FREE
plan_name = "Pro" if is_pro else "Free"

# === DISPLAY ===
col1, col2 = st.columns(2)
with col1:
    if is_pro:
        expires = datetime.fromtimestamp(plan["expires"]).strftime("%b %d, %Y")
        st.success(f"Pro Active · Expires: {expires}")
    else:
        st.info(f"Free · Max {MAX_CHARS_FREE:,} chars")
        if plan.get("error"):
            st.warning(plan["error"])
with col2:
    if not is_pro:
        st.markdown("[Upgrade to Pro →](https://www.mindscopeai.net//pricing-plans)")

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.write(f"**Plan:** {plan_name} | **Firebase OK:** {'Yes' if HAS_FIREBASE else 'No'}")
    language = st.selectbox("Language", ["English", "Hindi", "Polish", "Spanish", "Other"])
    region = st.selectbox("Region", ["India", "Poland", "US", "UK", "Global"])

# Main
col_in, col_out = st.columns(2)

with col_in:
    st.subheader("📝 Input Draft")
    sample = textwrap.dedent("""\
    As an AI language model, I'd be happy to help. Here's your article:
    In today's digital age, it's crucial to understand climate change. Not only rising seas, but also extreme weather play a crucial role.
    """)
    text = st.text_area(
    "Paste your article draft here",
    value=sample,
    height=350,
    label_visibility="collapsed"
)

with col_out:
    st.subheader("Cleaned & QC Results")

    # Initialize data as None
    data = None

    if st.button("Run CleanCopy", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Paste text first!")
        else:
            with st.spinner("Analyzing for AI leaks..."):
                data = local_qc(text)
                doc_len = len(text)

                if doc_len > max_chars:
                    st.error(f"Document too long ({doc_len:,} chars) for {plan_name} plan.")
                elif is_pro and HAS_GENAI:
                    g = gemini_clean(data["clean_text"], language, region, max_chars)
                    if g["ok"]:
                        data["clean_text"] = g["clean_text"]
                        st.info(f"Gemini polish applied in {g['elapsed']:.1f}s")

                # === AUTO-SCROLL ===
                components.html("<script>window.scrollTo(0, document.body.scrollHeight);</script>", height=0)

    # === SHOW RESULTS ONLY IF DATA EXISTS ===
    if data is not None:
        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("AI Risk", f"{data['ai_probability_score']}%")
        c2.metric("Severity", data['severity_score'])
        c3.metric("Plan", plan_name)

        # Leaks
        if data["prompt_leaks"]:
            st.error(f"{len(data['prompt_leaks'])} Prompt Leaks Found")
            for l in data["prompt_leaks"]:
                st.markdown(f"**Leak:** `{l['snippet'][:100]}...`  \n**Fix:** {l['fix']}")
                st.markdown("---")

        # Style Flags
        if data["other_risks"]:
            st.warning(f"{len(data['other_risks'])} AI Style Flags")
            for r in data["other_risks"][:3]:
                st.markdown(f"• **{r['snippet']}** → {r['reason']} ({r['fix']})")

        # Clean Text
        st.success("Clean Text Ready (Copy & Publish)")
        st.text_area(
            "Cleaned article (ready to copy)",
            value=data["clean_text"],
            height=400,
            label_visibility="collapsed"
        )

        # Debug
        with st.expander("Debug"):
            st.json({
                "Firebase": HAS_FIREBASE,
                "Gemini": HAS_GENAI,
                "Chars": doc_len,
                "Pro": is_pro,
                "Source": data.get("_meta", {}).get("source")
            })
    else:
        st.info("Click **Run CleanCopy** to analyze your text.")
# Footer
st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="clean")

# CLEAN CSS — NO COMMENTS
st.markdown("""
<style>
    .main { padding: 2rem; max-width: 1200px; margin: auto; overflow-y: auto; }
    .stButton > button { background: #1976D2; color: white; font-weight: bold; }
    .metric { background: #f5f5f5; padding: 1rem; border-radius: 10px; text-align: center; }
    .pro { background: #FFD700; color: #000; padding: 0.3rem 0.6rem; border-radius: 20px; font-weight: bold; }
    h1 { color: #1976D2; text-align: center; }
    .stButton > button:focus { scroll-margin-top: 100px; }
</style>
""", unsafe_allow_html=True)
st.markdown("*CleanCopy: AI-proof your copy. No domains needed – powered by Streamlit.*")