import os
import io
import re
import pickle
import random
import pathlib

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = pathlib.Path(__file__).parent
MODEL_PATH = ROOT / "model" / "model.pkl"
VEC_PATH   = ROOT / "model" / "vectorizer.pkl"

# ---------------------------------------------------------------------------
# Page config  (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake Job Analyzer AI",
    page_icon="🕵️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inlined utilities  (no external imports – works on every host)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text


SCAM_KEYWORDS = [
    "payment", "fee", "registration", "whatsapp", "urgent",
    "earn money", "no experience", "work from home", "guaranteed income",
    "wire transfer", "western union", "money order", "upfront",
    "investment required", "send money", "training fee",
    "background check fee", "unlimited earnings", "passive income",
    "be your own boss", "free laptop", "free iphone",
    "data entry", "copy paste", "part time from home",
]

KEYWORD_DESCRIPTIONS = {
    "payment":             "Requests payment from applicant",
    "fee":                 "Mentions fees or charges",
    "registration":        "Requires registration fee",
    "whatsapp":            "Uses WhatsApp for contact (unusual for legit jobs)",
    "urgent":              "Creates false urgency",
    "earn money":          "Vague earning promises",
    "no experience":       "Claims no experience needed (suspicious if paired with high pay)",
    "work from home":      "Unverified work-from-home offer",
    "guaranteed income":   "Promises guaranteed income",
    "wire transfer":       "Mentions wire transfer",
    "western union":       "Mentions Western Union payment",
    "money order":         "Requests money order",
    "upfront":             "Requires upfront payment",
    "investment required": "Requires financial investment",
    "send money":          "Asks to send money",
    "training fee":        "Charges training fee",
    "background check fee":"Charges for background check",
    "unlimited earnings":  "Claims unlimited earnings",
    "passive income":      "Promises passive income",
    "be your own boss":    "Uses 'be your own boss' pitch",
    "free laptop":         "Promises free laptop",
    "free iphone":         "Promises free iPhone",
    "data entry":          "Simple data entry tasks (often low-quality jobs)",
    "copy paste":          "Copy-paste tasks (common in scams)",
    "part time from home": "Unverified part-time work-from-home offer",
}


def calculate_risk(text: str):
    text_lower = text.lower()
    found      = [kw for kw in SCAM_KEYWORDS if kw in text_lower]
    score      = min(len(found) * 12, 100)
    reasons    = [KEYWORD_DESCRIPTIONS[kw] for kw in found]
    return score, reasons


# ---------------------------------------------------------------------------
# Training data  (fully inline – no CSV / external files needed)
# ---------------------------------------------------------------------------
_REAL_JOBS = [
    "Software Engineer at Google. Looking for an experienced engineer to design and build next-gen systems. 3+ years Python or Java required. Strong CS fundamentals, distributed systems experience.",
    "Marketing Manager - ABC Corp. Creative, data-driven marketer to lead campaigns. Manages social media, plans campaigns, analyses performance. Bachelor's degree required. Competitive salary and benefits.",
    "Data Scientist - FinTech Startup. Build predictive models for fraud detection and analytics. Skills: Python, SQL, machine learning. Remote-friendly, health insurance.",
    "Registered Nurse - City Hospital. Valid nursing license and 2+ years clinical experience required. Full-time and part-time. Excellent benefits.",
    "Product Manager - SaaS. Own the product roadmap. Work with engineering and design. 5+ years PM experience. Base + equity.",
    "Accountant - mid-size firm. Manage financial records, prepare tax returns, conduct audits. CPA preferred. Monday-Friday.",
    "UI/UX Designer. Strong portfolio required. Wireframes, prototypes, Figma. Salary $85k-$105k.",
    "DevOps Engineer. CI/CD pipelines, AWS, Kubernetes, Terraform. Comprehensive benefits including 401k.",
    "Sales Representative - B2B Software. Drive new business, manage SMB accounts. Base + commission. Training provided.",
    "Content Writer - Digital Agency. SEO-optimised blog posts for tech and finance clients. Portfolio required. Competitive per-word rate.",
    "Operations Manager - Logistics. Oversee warehouse operations, manage staff. 5+ years experience. Salary commensurate.",
    "Cybersecurity Analyst. Monitor incidents, vulnerability assessments. CISSP or CEH preferred.",
    "HR Specialist. Recruitment, onboarding, employee relations. SHRM preferred.",
    "Full Stack Developer - React/Node. Build and maintain web apps. Remote-friendly.",
    "Financial Analyst - Investment Bank. Build financial models. Excel and Bloomberg. CFA a plus.",
    "Customer Success Manager. Ensure client satisfaction and retention. 3+ years SaaS required.",
    "Electrical Engineer - Manufacturing. Design electrical systems. PE license preferred. On-site.",
    "Elementary School Teacher. Valid teaching certificate required. Small class sizes.",
    "Business Analyst - Consulting. Document requirements, improve processes. Up to 30% travel.",
    "Graphic Designer - Advertising. Adobe Creative Suite. 3+ years agency experience.",
    "Java Backend Developer. Spring Boot, Kafka, AWS. Remote work allowed.",
    "Research Scientist - AI Lab. PhD preferred. Top-tier compensation.",
    "Pharmacy Technician. State certification required. Flexible shifts.",
    "Civil Engineer. PE license, 5+ years. Strong project management.",
    "Social Media Manager. 2+ years experience. Hybrid schedule.",
]

_FAKE_JOBS = [
    "URGENT HIRING!!! Earn $5000 a week from home!!! No experience required!!! Send registration fee of $50. WhatsApp us now! Unlimited earnings guaranteed!!!",
    "Make money online from home. Copy paste job, earn $200 per day. No experience needed. Pay registration fee of $99 and start earning immediately. Guaranteed income every week!",
    "Data entry work from home. Earn $300-$500 daily. No experience needed. Send $30 training fee via Western Union to receive work kit.",
    "Part time work from home! Earn $150 per day filling simple online forms. Be your own boss! Payment via WhatsApp transfer. Urgent vacancies. No interview.",
    "WORK FROM HOME - Earn $1000 daily. Passive income. Free laptop and iPhone after registration. Pay $250 deposit refundable after 30 days. WhatsApp only.",
    "Online job! Earn $500/day working 2 hours from home. No experience. Send $75 registration fee via money order. Guaranteed payment every Monday.",
    "Home based job. Copy paste simple text. Earn $400 per day. Investment required: $199 to activate account. Be your own boss. Earn unlimited. Start today!",
    "Urgently needed: 500 workers for online data entry. Earn $50 per hour from home. No experience. Pay $49 joining fee. WhatsApp interview only.",
    "Make $3000 weekly from home. Part time from home job. Send $80 for training materials. Earn money without any investment. Guaranteed income.",
    "Online typing job. Earn $200-$500 per day. No experience. Pay $59 upfront for starter kit via wire transfer. Unlimited earnings potential.",
    "Earn money from your phone! Like and share posts on social media. Earn $100 per task. Registration fee $30. Payment guaranteed daily.",
    "GOVT APPROVED HOME JOB! Earn $2000 weekly. Fill forms from home. Investment required $149. No boss, no interview. WhatsApp for registration.",
    "Freelance data processing. Earn $350 per day copying files. Training fee $99 via Western Union. No experience. Urgent: 200 seats only!",
    "Work from home - watch videos, click ads. Earn $200/hour. Registration fee $25. Unlimited earnings. Guaranteed payment.",
    "Online assistant needed urgently. Work from home, earn $600/day. No experience, no interview. Pay $89 activation fee. WhatsApp contact.",
    "Passive income from home! Earn $5000/month doing nothing. Just refer friends. Registration fee $50. Guaranteed returns!",
    "Data entry jobs - no experience! Earn $1000 per week from home. Send $79 registration fee via wire transfer.",
    "Online work from home job! Earn $300 daily. Free training after $99 registration fee. WhatsApp for immediate joining.",
    "Home-based job. Copy paste simple text. Earn $250 per hour. Send $45 fee to unlock account. Earn money guaranteed every day.",
    "LAST 5 SEATS! Earn $2500 weekly from home. No experience. Investment $199. Be your own boss. WhatsApp interview. Earn unlimited!",
    "Online mystery shopper required urgently! Earn $500 per task. Background check fee $60 required. WhatsApp for details.",
    "Social media evaluator - work from home! Earn $400 daily. Check Facebook posts. Registration fee $35. Guaranteed weekly payment.",
    "Amazon home packer job! Pack orders from home. Earn $50/box. Send $99 kit fee. Work from home. WhatsApp: immediate start!",
    "Earn money from home - survey jobs! Earn $200 per survey. Registration fee $29. Part time from home. Guaranteed payment!",
    "YouTube video rating job. Earn $100 per video. No experience. Pay $55 activation fee. Earn unlimited. WhatsApp registration!!",
]


def _build_training_df():
    rows = [(t, 0) for t in _REAL_JOBS] + [(t, 1) for t in _FAKE_JOBS]
    augmented = []
    rng  = random.Random(42)
    pfx  = ["HIRING NOW: ", "JOB ALERT: ", "VACANCY: ", ""]
    sfx  = [" Apply now!", " Limited seats.", " Contact us today.", ""]
    for text, label in rows:
        augmented.append((text, label))
        for _ in range(5):
            augmented.append((rng.choice(pfx) + text + rng.choice(sfx), label))
    rng.shuffle(augmented)
    return pd.DataFrame(augmented, columns=["description", "fraudulent"])


# ---------------------------------------------------------------------------
# Model – load from disk or train fresh
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="🤖 Training AI model on first launch…")
def load_model():
    if MODEL_PATH.exists() and VEC_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            with open(VEC_PATH, "rb") as f:
                vec = pickle.load(f)
            return model, vec
        except Exception:
            pass  # corrupted – retrain

    df    = _build_training_df()
    X     = df["description"].astype(str)
    y     = df["fraudulent"].astype(int)
    vec   = TfidfVectorizer(stop_words="english", max_features=10_000)
    X_vec = vec.fit_transform(X)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_vec, y)

    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(VEC_PATH, "wb") as f:
            pickle.dump(vec, f)
    except Exception:
        pass  # read-only fs on some cloud hosts – that's fine

    return model, vec


# ---------------------------------------------------------------------------
# Voice transcription  (graceful fallback if google unreachable)
# ---------------------------------------------------------------------------
def transcribe_audio(audio_bytes: bytes) -> str:
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# AI response builder
# ---------------------------------------------------------------------------
def build_ai_response(pred, fake_prob, real_prob, risk_score, reasons):
    if pred == 1:
        opening  = "🚨 **This job posting appears to be FRAUDULENT.**"
        tone     = "very high — strong scam signals detected" if fake_prob > 75 else "notable — treat with caution"
        ml_line  = f"My ML model is **{fake_prob:.1f}% confident** this is a fake job, which is {tone}."
    else:
        opening  = "✅ **This job posting appears to be GENUINE.**"
        conf_str = "very high confidence" if real_prob > 80 else "reasonably confident"
        ml_line  = f"My ML model is **{real_prob:.1f}% confident** this is a legitimate job posting — {conf_str}."

    risk_label = (
        "🔴 **HIGH** — serious red flags present"   if risk_score > 50 else
        "🟠 **MEDIUM** — some suspicious elements"  if risk_score > 25 else
        "🟡 **LOW** — minor caution advised"         if risk_score > 0  else
        "🟢 **NONE** — no scam keywords detected"
    )

    out  = f"{opening}\n\n"
    out += f"**ML Analysis:** {ml_line}\n\n"
    out += f"**Risk Score:** {risk_score}/100 — {risk_label}\n\n"

    if reasons:
        out += "**⚠️ Suspicious Signals Detected:**\n"
        for r in reasons:
            out += f"- {r}\n"
        out += "\n"

    if pred == 1:
        out += (
            "**💡 Recommendation:** Do **not** apply. Avoid sharing personal information, "
            "paying any fees, or contacting via WhatsApp. Report this posting to the job platform."
        )
    elif risk_score > 0:
        out += (
            "**💡 Recommendation:** Looks real, but minor flags detected. "
            "Verify the company independently before proceeding."
        )
    else:
        out += (
            "**💡 Recommendation:** Looks like a legitimate opportunity. "
            "Still research the company before sharing sensitive personal details."
        )
    return out


def analyse(text: str) -> str:
    cleaned  = clean_text(text)
    vec_out  = vectorizer.transform([cleaned])
    pred     = model.predict(vec_out)[0]
    proba    = model.predict_proba(vec_out)[0]
    risk, rs = calculate_risk(text)
    return build_ai_response(pred, proba[1] * 100, proba[0] * 100, risk, rs)


# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------
model, vectorizer = load_model()

# ---------------------------------------------------------------------------
# Sample jobs
# ---------------------------------------------------------------------------
SAMPLE_FAKE = (
    "URGENT HIRING!!! Earn $5000 a week from home!!! No experience required!!! "
    "Send registration fee of $50 to get started. WhatsApp us now! "
    "Work from home, unlimited earnings guaranteed!!!"
)
SAMPLE_REAL = (
    "Software Engineer – Full Stack (React / Node.js). We are building the next generation "
    "of our SaaS platform and looking for an experienced full-stack engineer. You will design "
    "and implement new features end-to-end, collaborate with product and design, and ensure "
    "high availability. Requirements: 3+ years experience with React and Node.js, familiarity "
    "with PostgreSQL and REST APIs. Remote-friendly. Competitive salary, equity, and benefits."
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hi! I'm your **Fake Job Analyzer AI**.\n\n"
                "I detect whether any job posting is **genuine or fraudulent** using ML + NLP.\n\n"
                "**How to use me:**\n"
                "- 💬 **Type** a job description in the chat box below\n"
                "- 🎤 **Record** using the voice recorder (then click Analyse)\n"
                "- 📋 Try a **sample job** from the sidebar\n\n"
                "Paste or speak any job posting — I'll analyse it instantly!"
            ),
        }
    ]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🕵️ Fake Job Analyzer AI")
    st.markdown("*AI-powered job fraud detection*")
    st.divider()

    st.markdown("### 📋 Try Sample Jobs")
    if st.button("🚨 Load Fake Job Sample", use_container_width=True):
        st.session_state["inject_text"] = SAMPLE_FAKE
    if st.button("✅ Load Real Job Sample", use_container_width=True):
        st.session_state["inject_text"] = SAMPLE_REAL

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared! Paste or speak any job description and I'll analyse it.",
            }
        ]
        st.rerun()

    st.divider()
    st.markdown("### ℹ️ How It Works")
    st.markdown(
        "1. **Type or record** a job description\n"
        "2. ML model classifies using TF-IDF + Logistic Regression\n"
        "3. Risk engine scans 25 scam keywords\n"
        "4. Get a verdict with full reasoning\n"
    )
    st.divider()
    st.markdown("### ⚠️ Common Scam Signals")
    st.markdown(
        "- 💸 Upfront fees or registration\n"
        "- 📲 WhatsApp-only contact\n"
        "- 🎁 Free gadgets promised\n"
        "- ⚡ No experience + high pay\n"
        "- 💰 Guaranteed income claims\n"
        "- 🔴 'Urgent' / limited seats language\n"
    )
    st.divider()
    st.caption("Made by Aishwarya Lala · SVPCET")

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.markdown("## 🕵️ Fake Job Analyzer AI")
st.markdown(
    "**Type** or 🎤 **record** any job description — "
    "I'll instantly tell you if it's real or fake."
)

# Voice recorder
with st.expander("🎤 Voice Input — Record a Job Description", expanded=False):
    st.caption("Click the microphone, speak the job description, then click **Analyse Recording**.")
    audio_value = st.audio_input("Record job description")
    if audio_value is not None:
        if st.button("🔍 Analyse Recording", type="primary", use_container_width=True):
            with st.spinner("Transcribing audio…"):
                transcript = transcribe_audio(audio_value.getvalue())
            if transcript:
                st.success(f"**Transcribed:** {transcript}")
                st.session_state["inject_text"] = transcript
                st.rerun()
            else:
                st.warning(
                    "Could not transcribe audio. "
                    "Please speak clearly or type the description manually."
                )

st.divider()

# Render chat history
for msg in st.session_state.messages:
    avatar = "🕵️" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Inject from sidebar / voice
if "inject_text" in st.session_state:
    inject = st.session_state.pop("inject_text")
    with st.chat_message("user", avatar="👤"):
        st.markdown(inject)
    st.session_state.messages.append({"role": "user", "content": inject})
    with st.chat_message("assistant", avatar="🕵️"):
        with st.spinner("Analysing…"):
            response = analyse(inject)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Chat input
if prompt := st.chat_input("Paste a job description here…"):
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant", avatar="🕵️"):
        with st.spinner("Analysing…"):
            response = analyse(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
