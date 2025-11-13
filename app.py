import os
import json
import textwrap

import streamlit as st

# Try to import Gemini SDK
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

API_KEY_ENV_VAR = "GEMINI_API_KEY"
MODEL_NAME = "gemini-2.5-flash"

# Initialize key holder
GEMINI_API_KEY = None

# ===========================
# Configure Gemini
# ===========================
if HAS_GENAI:
    GEMINI_API_KEY = os.getenv(API_KEY_ENV_VAR)
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
else:
    GEMINI_API_KEY = None


# ===========================
# System instructions
# ===========================
SYSTEM_INSTRUCTIONS = """
You are an AI Editorial Quality Control engine.

Your job:
Given:
- raw text content
- language
- target region

You MUST return a STRICT JSON object with the following structure:

{
  "prompt_leaks": [
    {
      "snippet": "string",
      "reason": "string",
      "suggested_fix": "string"
    }
  ],
  "tone_issues": [
    {
      "snippet": "string",
      "reason": "string",
      "suggested_fix": "string"
    }
  ],
  "region_mismatches": [
    {
      "snippet": "string",
      "reason": "string",
      "suggested_fix": "string"
    }
  ],
  "other_risks": [
    {
      "snippet": "string",
      "reason": "string",
      "suggested_fix": "string"
    }
  ],
  "ai_probability_score": 0,
  "severity_score": 0,
  "clean_text": "string"
}

Rules:
- Return ONLY the JSON object. No explanations, no markdown, no comments.
- Each list can be empty.
- "ai_probability_score" is an integer 0–100.
- "severity_score" is an integer 0–10.
- "clean_text" MUST be the polished, publication-ready version.
"""


# ===========================
# Gemini call helper
# ===========================
def run_qc(text: str, language: str, region: str) -> dict:
    """
    Send the QC request to Gemini, return parsed JSON or an error dict.
    """
    if not HAS_GENAI:
        return {"error": "google-generativeai not installed. Run: pip install google-generativeai"}

    if not GEMINI_API_KEY:
        return {"error": f"{API_KEY_ENV_VAR} is not set in the environment."}

    model = genai.GenerativeModel(MODEL_NAME)

    user_prompt = f"""
Content language: {language}
Target region: {region}

Content to review:

```text
{text}
```"""
    full_prompt = SYSTEM_INSTRUCTIONS + "\n" + user_prompt
    response = model.generate_content(
        [SYSTEM_INSTRUCTIONS, user_prompt]
    )

    raw = (getattr(response, "text", "") or "").strip()

    # Try to clean up JSON if wrapped in ```json ... ```
    try:
        if raw.startswith("```"):
            # remove ```...``` wrapper
            raw = raw.strip("`")
            idx = raw.find("{")
            if idx != -1:
                raw = raw[idx:]
        data = json.loads(raw)
    except Exception as e:
        data = {
            "error": f"JSON parsing failed: {e}",
            "raw_response": raw,
        }

    return data    

st.set_page_config(page_title="Editorial Guard MVP", layout="wide")

st.title("🛡️ Editorial Guard – MVP")
st.caption("Paste AI-written content → detect leaks, tone issues, region mismatches, and get a cleaned version.")

# Sidebar
st.sidebar.header("Settings")

language = st.sidebar.selectbox(
    "Language",
    ["English", "Urdu", "Hindi", "Spanish", "Other"],
    index=0,
)

region = st.sidebar.selectbox(
    "Target Region",
    ["US", "UK", "India", "Pakistan", "EU", "Global"],
    index=3,
)

st.sidebar.markdown("---")
st.sidebar.write("⚙️ Model: Gemini 1.5 Flash")

if not HAS_GENAI:
    st.sidebar.error("❌ google-generativeai not installed.")
elif not GEMINI_API_KEY:
    st.sidebar.error(f"❌ {API_KEY_ENV_VAR} not set.")

col_input, col_output = st.columns(2)

with col_input:
    st.subheader("1️⃣ Input Content")

    sample_text = textwrap.dedent(
        """\
        Write a 500-word editorial about inflation in Pakistan for a national newspaper.
        Make it neutral and informative.

        Do you want me to do that next?
        """
    )

    text = st.text_area(
        "Paste your article / draft here:",
        value=sample_text,
        height=350,
    )

    run_button = st.button(
        "🔍 Run QC",
        type="primary",
        use_container_width=True,
    )

with col_output:
    st.subheader("2️⃣ QC Results")

    summary_placeholder = st.empty()
    details_placeholder = st.empty()
    cleaned_placeholder = st.empty()

    if run_button:
        if not text.strip():
            st.warning("Please paste some text first.")
        else:
            with st.spinner("Running QC…"):
                data = run_qc(text, language, region)

            # Hard error (config / JSON failure and no partial results)
            if "error" in data and not any(
                data.get(k) for k in ["prompt_leaks", "tone_issues", "region_mismatches", "other_risks"]
            ):
                st.error("Error: " + data["error"])
                raw_resp = data.get("raw_response")
                if raw_resp:
                    with st.expander("Raw model output"):
                        st.code(raw_resp, language="json")
            else:
                # ----------- Summary -----------
                sev = data.get("severity_score", "N/A")
                aip = data.get("ai_probability_score", "N/A")

                with summary_placeholder.container():
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("🚨 Severity (0–10)", sev)
                    with c2:
                        st.metric("🤖 AI Probability (0–100%)", aip)

                # ----------- Issues -----------
                def render_section(title: str, key: str):
                    issues = data.get(key) or []
                    st.markdown(f"#### {title}")
                    if not issues:
                        st.success(f"No {title.lower()} detected.")
                        return
                    for issue in issues:
                        snippet = issue.get("snippet", "")
                        reason = issue.get("reason", "")
                        fix = issue.get("suggested_fix", "")
                        if snippet:
                            st.markdown(f"**Snippet:** `{snippet}`")
                        if reason:
                            st.markdown(f"**Reason:** {reason}")
                        if fix:
                            st.markdown(f"**Suggested fix:** {fix}")
                        st.markdown("---")

                with details_placeholder.container():
                    st.markdown("### 🔎 Issues Found")
                    render_section("Prompt Leaks", "prompt_leaks")
                    render_section("Tone Issues", "tone_issues")
                    render_section("Region Mismatches", "region_mismatches")
                    render_section("Other Risks", "other_risks")

                # ----------- Cleaned Text -----------
                cleaned = data.get("clean_text", "")

                with cleaned_placeholder.container():
                    st.markdown("### ✅ Cleaned, Ready-to-Publish Text")
                    st.text_area(
                        "Copy this text:",
                        value=cleaned,
                        height=250,
                    )
                    st.caption("Unwanted AI meta-talk removed, tone aligned, region adapted where possible.")

