import streamlit as st
import io


def render_single_analysis():
    st.markdown("## 🕵️ Fake Job Analyzer AI - Single Job")
    st.markdown(
        "**Type** or 🎤 **record** any job description — "
        "I'll instantly tell you if it's real or fake."
    )
    
    # Initialize session state for messages if not exists
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

# Helpers
def transcribe_audio(audio_bytes: bytes) -> str:
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except Exception:
        return ""

def analyse(text: str) -> str:
    from services.model_service import load_model
    # We call the globally loaded model and vectorizer
    model, vectorizer = load_model()
    
    from utils.feature_extractor import clean_text
    from utils.risk_calculator import calculate_risk
    
    cleaned = clean_text(text)
    vec_out = vectorizer.transform([cleaned])
    pred = model.predict(vec_out)[0]
    proba = model.predict_proba(vec_out)[0]
    
    risk, rs = calculate_risk(text)
    
    return build_ai_response(pred, proba[1] * 100, proba[0] * 100, risk, rs)

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
