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
# Page config  (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake Job Analyzer AI",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.styles import apply_custom_css
from ui.single_analysis import render_single_analysis
from ui.batch_analysis import render_batch_analysis

# Apply global premium CSS
apply_custom_css()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = pathlib.Path(__file__).parent
MODEL_PATH = ROOT / "model" / "model.pkl"
VEC_PATH   = ROOT / "model" / "vectorizer.pkl"

from services.model_service import load_model
# Force load model to cache
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
        "3. Risk engine scans for granular scam indicators\n"
        "4. **Upload folders/files** in Batch Analysis mode\n"
    )
    st.divider()
    st.caption("Made by Aishwarya Lala · SVPCET")

# ---------------------------------------------------------------------------
# Tabs for Main UI
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["💬 Single Job Analysis", "📂 Batch Job Analysis"])

with tab1:
    render_single_analysis()

with tab2:
    render_batch_analysis()
