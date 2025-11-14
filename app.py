# app.py — CleanCopy: Firebase + Gemini + Multi-Strategy Scroll
import os
import re
import time
import json
import textwrap
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

# ----------------------------------------------------------------------
# PAGE CONFIG - MUST BE FIRST
# ----------------------------------------------------------------------
st.set_page_config(page_title="CleanCopy", layout="wide", page_icon="🧹")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
APP_NAME = "CleanCopy"
MODEL_NAME = "gemini-1.5-flash"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MAX_CHARS_FREE = 10_000
MAX_CHARS_PRO  = 50_000
GEMINI_HARD_LIMIT_CHARS = 30_000

# SCROLL STRATEGY: 'auto' | 'query_params' | 'aggressive'
SCROLL_MODE = os.getenv("SCROLL_MODE", "auto")

# ----------------------------------------------------------------------
# FIREBASE (SAFE INITIALISATION)
# ----------------------------------------------------------------------
FIRESTORE_DB = None
HAS_FIREBASE = False
FIREBASE_ERROR = None

try:
    import firebase_admin
    from firebase_admin import firestore, credentials

    cred_json = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if cred_json:
        cred = credentials.Certificate(json.loads(cred_json))
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        FIRESTORE_DB = firestore.client()
        HAS_FIREBASE = True
except Exception as e:
    FIREBASE_ERROR = str(e)
    HAS_FIREBASE = False

# ----------------------------------------------------------------------
# GEMINI
# ----------------------------------------------------------------------
HAS_GENAI = False
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        HAS_GENAI = True
except Exception:
    pass

# ----------------------------------------------------------------------
# AI DETECTION PATTERNS
# ----------------------------------------------------------------------
AI_PROMPT_PATTERNS = [
    r"\bas an a[i1l] language model\b", r"\bi am an a[i1l]\b", r"\bi'?m an a[i1l]\b",
    r"\blanguage model\b", r"\$\$ system prompt[:\s]\$\$", r"\bi'?d be happy to assist\b",
    r"\bi'?m here to help\b", r"\bif you want, i can also\b", r"\bdo you want me to\b",
    r"\bhere'?s your\b",
]

AI_STYLE_KEYWORDS = [
    "in the world of", "in today's digital age", "it is crucial to",
    "let's delve into", "dive deep", "not only", "but also", "play a crucial role"
]

FORMAL_GREETINGS = ["i hope this email finds you well"]

# ----------------------------------------------------------------------
# UTILS
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# GEMINI CLEAN (Pro only)
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# FIREBASE AUTH
# ----------------------------------------------------------------------
def get_user_plan(email_or_token):
    if not HAS_FIREBASE or not FIRESTORE_DB:
        return {"is_pro": False, "error": "Firebase not connected"}

    token = str(email_or_token).strip()
    if not token or token == "None":
        return {"is_pro": False}

    try:
        # 1. Token as document ID
        doc_ref = FIRESTORE_DB.collection("pro_access").document(token)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            if data.get("expires", 0) > time.time():
                return {"is_pro": True, "expires": data["expires"], "email": data.get("email")}

        # 2. Email lookup
        if "@" in token:
            query = FIRESTORE_DB.collection("pro_access").where("email", "==", token).limit(1).stream()
            for d in query:
                data = d.to_dict()
                if data.get("expires", 0) > time.time():
                    return {"is_pro": True, "expires": data["expires"], "email": data["email"], "token": d.id}
    except Exception as e:
        return {"is_pro": False, "error": str(e)}

    return {"is_pro": False}

# ----------------------------------------------------------------------
# SCROLL INJECTION STRATEGIES
# ----------------------------------------------------------------------
def trigger_scroll_to_results():
    """Pure Streamlit scroll using container manipulation"""
    # Use st.empty() placeholder to force re-render at bottom
    st.markdown(
        """
        <style>
        @keyframes scrollToBottom {
            to { scroll-behavior: smooth; }
        }
        .scroll-target {
            animation: scrollToBottom 0.1s;
            scroll-margin-top: 100px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
def inject_scroll_auto():
    """Not used - pure Python approach preferred"""
    pass

def inject_scroll_aggressive():
    """Not used - pure Python approach preferred"""
    pass

# ----------------------------------------------------------------------
# PAGE CONFIG + CSS
# ----------------------------------------------------------------------
# Enhanced CSS with scroll forcing
st.markdown("""
<style>
    .main { padding: 2rem; max-width: 1200px; margin: auto; }
    .stButton > button { background:#1976D2; color:white; font-weight:bold; }
    .metric { background:#f5f5f5; padding:1rem; border-radius:10px; text-align:center; }
    h1 { color:#1976D2; text-align:center; }
    
    /* Force scrollable containers */
    html, body, [data-testid="stApp"] {
        overflow-y: auto !important;
        scroll-behavior: smooth !important;
    }
    
    section[data-testid="stVerticalBlock"] {
        overflow-y: auto !important;
    }
    
    /* Scroll anchor */
    #results {
        scroll-margin-top: 20px;
        padding-top: 10px;
    }
    
    /* Ensure columns are scrollable */
    [data-testid="column"] {
        overflow-y: visible !important;
    }
</style>
""", unsafe_allow_html=True)

# Show Firebase error if exists
if FIREBASE_ERROR:
    st.sidebar.warning(f"Firebase: {FIREBASE_ERROR}")

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.title(f"🧹 {APP_NAME}")
st.markdown("*AI wrote it. We fix it. For journalists worldwide.*")

# ----------------------------------------------------------------------
# AUTH + PLAN DISPLAY
# ----------------------------------------------------------------------
qp = st.query_params
token = qp.get("token")
email_input = st.text_input(
    "Email (optional for Pro)", placeholder="you@example.com",
    help="Enter to check Pro status"
)
email = email_input.strip() or ""
token = token or ""

plan = get_user_plan(token or email)
is_pro = plan.get("is_pro", False)
max_chars = MAX_CHARS_PRO if is_pro else MAX_CHARS_FREE
plan_name = "Pro" if is_pro else "Free"

col1, col2 = st.columns(2)
with col1:
    if is_pro:
        exp = datetime.fromtimestamp(plan["expires"]).strftime("%b %d, %Y")
        st.success(f"✅ Pro Active · Expires: {exp}")
    else:
        st.info(f"🆓 Free · Max {MAX_CHARS_FREE:,} chars")
        if plan.get("error"):
            st.warning(plan["error"])
with col2:
    if not is_pro:
        st.markdown("[⬆️ Upgrade to Pro →](https://www.mindscopeai.net/pricing-plans)")

# ----------------------------------------------------------------------
# SIDEBAR SETTINGS
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"**Plan:** {plan_name} | **Firebase:** {'✅' if HAS_FIREBASE else '❌'}")
    st.write(f"**Scroll Mode:** {SCROLL_MODE}")
    language = st.selectbox("Language", ["English", "Hindi", "Polish", "Spanish", "Other"])
    region   = st.selectbox("Region",   ["India", "Poland", "US", "UK", "Global"])
    
    st.markdown("---")
    st.markdown("**Scroll Debug**")
    if st.button("Test Auto Scroll"):
        st.session_state.scroll_trigger = True
        st.rerun()

# ----------------------------------------------------------------------
# MAIN COLUMNS
# ----------------------------------------------------------------------
col_in, col_out = st.columns(2)

# ---------- INPUT ----------
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

# ---------- OUTPUT ----------
with col_out:
    st.subheader("✨ Cleaned & QC Results")
    
    # NUCLEAR OPTION: Query params mode
    if SCROLL_MODE == "query_params":
        if qp.get("view") == "results":
            st.session_state.show_results = True
            # Don't clear immediately - let it render first
    
    show_results = st.session_state.get("show_results", False)
    
    if st.button("🚀 Run CleanCopy", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("⚠️ Paste text first!")
        else:
            with st.spinner("🔄 Analyzing..."):
                data = local_qc(text)
                doc_len = len(text)

                if doc_len > max_chars:
                    st.error(f"❌ Text exceeds {plan_name} limit ({doc_len:,} > {max_chars:,})")
                else:
                    if HAS_GENAI and is_pro:
                        g = gemini_clean(data["clean_text"], language, region, max_chars)
                        if g["ok"]:
                            data["clean_text"] = g["clean_text"]
                            st.info(f"✨ Gemini polish in {g['elapsed']:.1f}s")
                    
                    st.session_state.results = data
                    st.session_state.show_results = True
                    
                    # NUCLEAR: Use query params for guaranteed scroll
                    if SCROLL_MODE == "query_params":
                        qp["view"] = "results"
                    
                    st.rerun()

    # Display results FIRST, then clean params
    if show_results and "results" in st.session_state:
        data = st.session_state.results
        
        # Clear query params AFTER render
        if SCROLL_MODE == "query_params" and qp.get("view") == "results":
            # Use a separate flag to prevent infinite loop
            if not st.session_state.get("params_cleared"):
                st.session_state.params_cleared = True
            else:
                qp.clear()
                st.session_state.params_cleared = False
        
        # Visual separator and anchor
        st.markdown("---")
        st.markdown('<div class="scroll-target"></div>', unsafe_allow_html=True)
        st.markdown("## 📊 Analysis Results")
        
        # Add visual indicator that results are ready
        st.success("✅ Analysis Complete - Results Below")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            delta_color = "inverse" if data['ai_probability_score'] > 80 else "normal"
            st.metric("AI Risk", f"{data['ai_probability_score']}%", 
                     delta=f"Severity {data['severity_score']}/10")
        with c2:
            total_issues = len(data['prompt_leaks']) + len(data['other_risks'])
            st.metric("Issues Found", total_issues)
        with c3:
            st.metric("Plan", plan_name, delta="Active" if is_pro else "Free")

        if data["prompt_leaks"]:
            with st.expander(f"🚨 {len(data['prompt_leaks'])} Critical Leaks", expanded=True):
                for i, l in enumerate(data["prompt_leaks"], 1):
                    st.error(f"**Leak #{i}:** `{l['snippet'][:100]}...`")
                    st.caption(f"💡 **Fix:** {l['fix']}")
                    if i < len(data["prompt_leaks"]):
                        st.markdown("---")

        if data["other_risks"]:
            with st.expander(f"⚠️ {len(data['other_risks'])} Style Flags", expanded=False):
                for i, r in enumerate(data["other_risks"], 1):
                    st.warning(f"**Flag #{i}:** {r['snippet']}")
                    st.caption(f"📝 {r['reason']} → {r['fix']}")
                    if i < len(data["other_risks"]):
                        st.markdown("---")

        st.markdown("### ✅ Cleaned Output")
        cleaned = st.text_area(
            "Copy cleaned text below",
            value=data["clean_text"],
            height=350,
            label_visibility="collapsed",
            key="cleaned_output"
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Analyze New Text", use_container_width=True):
                st.session_state.show_results = False
                st.session_state.params_cleared = False
                if "results" in st.session_state:
                    del st.session_state.results
                qp.clear()
                st.rerun()
        with col_b:
            st.download_button(
                "💾 Download Clean Copy",
                data=cleaned,
                file_name=f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    else:
        st.info("👆 Click **Run CleanCopy** to analyze your text")

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown("---")
st.markdown(f"*{APP_NAME} — AI-proof your copy. Powered by Streamlit · Mode: {SCROLL_MODE}*")

# WIX IFRAME INTEGRATION INSTRUCTIONS
st.markdown("""
<details>
<summary><b>📦 Wix Embedding Instructions</b></summary>

Add this to your Wix page's **Custom Code** (Settings → Custom Code → Body End):

```html
<script>
window.addEventListener('message', (e) => {
    if (e.data.type === 'streamlit:scroll') {
        const iframe = document.querySelector('iframe[src*="streamlit"]');
        if (iframe) {
            iframe.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
    }
});
</script>
```

**Environment Variables for Streamlit Cloud:**
- `SCROLL_MODE=aggressive` (for stubborn iframes)
- `SCROLL_MODE=query_params` (most reliable)
- `SCROLL_MODE=auto` (default, balanced)

</details>
""", unsafe_allow_html=True)