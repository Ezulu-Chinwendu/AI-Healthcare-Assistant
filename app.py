import streamlit as st
from rag_pipeline import ask_medical_question

# ------------------------------
# PAGE CONFIG
# ------------------------------

st.set_page_config(
    page_title="MedAI — Medical Assistant",
    page_icon="🩺",
    layout="centered"
)

# ------------------------------
# CUSTOM STYLING
# ------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #F5F0E8 !important;
    font-family: 'DM Sans', sans-serif;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* Main container */
[data-testid="stAppViewContainer"] > .main > .block-container {
    max-width: 760px;
    padding: 0 24px 80px 24px !important;
}

/* ── Masthead ── */
.masthead {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 52px 0 36px;
    text-align: center;
}

.masthead-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #9C8B6E;
    margin-bottom: 14px;
}

.masthead-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(38px, 6vw, 56px);
    line-height: 1.08;
    color: #1C1712;
    letter-spacing: -0.02em;
}

.masthead-title em {
    font-style: italic;
    color: #7C6A4A;
}

.masthead-sub {
    font-size: 15px;
    color: #6B6150;
    margin-top: 14px;
    line-height: 1.6;
    max-width: 500px;
}

/* ── Divider ── */
.serif-rule {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 28px 0;
    color: #C9BC9F;
    font-size: 18px;
}
.serif-rule::before,
.serif-rule::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, transparent, #C9BC9F, transparent);
}

/* ── Info card ── */
.info-card {
    background: #FFFDF7;
    border: 1px solid #E2D9C5;
    border-left: 3px solid #B8A07A;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 28px;
}

.info-card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 15px;
    color: #1C1712;
    margin-bottom: 10px;
    letter-spacing: 0.01em;
}

.info-topics {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 14px;
}

.topic-pill {
    background: #F0EAD8;
    border: 1px solid #D9CEBC;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12.5px;
    color: #5A4E3A;
    font-weight: 500;
}

.info-disclaimer {
    font-size: 12px;
    color: #9C8B6E;
    border-top: 1px solid #E8E0CE;
    padding-top: 10px;
    margin-top: 4px;
    line-height: 1.5;
}

/* ── Name input area ── */
.name-card {
    background: #FFFDF7;
    border: 1px solid #E2D9C5;
    border-radius: 12px;
    padding: 28px 32px;
    text-align: center;
    margin-bottom: 28px;
}

.name-card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #1C1712;
    margin-bottom: 6px;
}

.name-card-sub {
    font-size: 14px;
    color: #7D6F5A;
    margin-bottom: 20px;
}

/* ── Streamlit input override ── */
[data-testid="stTextInput"] input {
    background: #FAF6EE !important;
    border: 1.5px solid #D4C9B0 !important;
    border-radius: 8px !important;
    color: #1C1712 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 10px 16px !important;
    text-align: center;
    transition: border-color 0.2s;
}

[data-testid="stTextInput"] input:focus {
    border-color: #B8A07A !important;
    box-shadow: 0 0 0 3px rgba(184,160,122,0.15) !important;
    outline: none !important;
}

/* ── Greeting banner ── */
.greeting-banner {
    background: linear-gradient(135deg, #2D2318 0%, #4A3728 100%);
    border-radius: 10px;
    padding: 16px 22px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 14px;
    animation: fadeSlideIn 0.5s ease;
}

.greeting-avatar {
    width: 40px;
    height: 40px;
    background: #B8A07A;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}

.greeting-text {
    color: #F5F0E8;
    font-size: 14px;
    line-height: 1.5;
}

.greeting-text strong {
    color: #D4B896;
    font-family: 'DM Serif Display', serif;
    font-size: 16px;
}

/* ── Chat messages ── */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-bottom: 24px;
}

.msg-row {
    display: flex;
    gap: 12px;
    animation: fadeSlideIn 0.35s ease;
}

.msg-row.user-row { justify-content: flex-end; }
.msg-row.bot-row  { justify-content: flex-start; }

.msg-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    margin-top: 2px;
}

.user-avatar { background: #2D2318; order: 2; }
.bot-avatar  { background: #B8A07A; }

.msg-bubble {
    max-width: 82%;
    padding: 13px 18px;
    border-radius: 16px;
    font-size: 14.5px;
    line-height: 1.65;
}

.user-bubble {
    background: #2D2318;
    color: #F5F0E8;
    border-bottom-right-radius: 4px;
}

.bot-bubble {
    background: #FFFDF7;
    border: 1px solid #E2D9C5;
    color: #2A201A;
    border-bottom-left-radius: 4px;
}

.msg-label {
    font-size: 10.5px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 5px;
    opacity: 0.55;
}

.user-bubble .msg-label { color: #D4B896; }
.bot-bubble  .msg-label { color: #9C8B6E; }

/* Source citation in bot message */
.source-line {
    margin-top: 10px;
    padding-top: 9px;
    border-top: 1px dashed #E2D9C5;
    font-size: 12px;
    color: #9C8B6E;
    font-style: italic;
}

/* ── Chat input override ── */
[data-testid="stChatInput"] {
    background: #FFFDF7 !important;
    border: 1.5px solid #D4C9B0 !important;
    border-radius: 12px !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #1C1712 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
}

[data-testid="stChatInput"] button {
    background: #2D2318 !important;
    border-radius: 8px !important;
    color: #F5F0E8 !important;
}

/* ── Spinner override ── */
[data-testid="stSpinner"] p {
    color: #7D6F5A !important;
    font-size: 14px !important;
    font-style: italic;
}

/* ── Success/info overrides ── */
[data-testid="stAlert"] {
    background: #FFFDF7 !important;
    border-color: #B8A07A !important;
    color: #3A2E22 !important;
    border-radius: 8px !important;
}

/* ── Animations ── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.masthead { animation: fadeSlideIn 0.6s ease; }

</style>
""", unsafe_allow_html=True)


# ------------------------------
# MASTHEAD
# ------------------------------

st.markdown("""
<div class="masthead">
    <div class="masthead-eyebrow">Powered by Medical AI</div>
    <div class="masthead-title">Your <em>personal</em><br>health guide</div>
    <p class="masthead-sub">Ask questions about conditions, symptoms, treatments,
    and medications — answered from trusted medical sources.</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------
# INFO CARD
# ------------------------------

st.markdown("""
<div class="info-card">
    <div class="info-card-title">What I can help with</div>
    <div class="info-topics">
        <span class="topic-pill">🦠 Diseases &amp; Conditions</span>
        <span class="topic-pill">🔍 Symptoms</span>
        <span class="topic-pill">💊 Medications</span>
        <span class="topic-pill">🏥 Treatments</span>
        <span class="topic-pill">🧪 Diagnostic Tests</span>
        <span class="topic-pill">🧬 Medical Procedures</span>
    </div>
    <div class="info-disclaimer">
        ⚠️ For educational purposes only. This assistant does not replace professional medical advice,
        diagnosis, or treatment. Always consult a qualified healthcare provider.
    </div>
</div>
""", unsafe_allow_html=True)


# ------------------------------
# SESSION STATE
# ------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "name" not in st.session_state:
    st.session_state.name = None


# ------------------------------
# ASK NAME
# ------------------------------

if st.session_state.name is None:
    st.markdown("""
    <div class="name-card">
        <div class="name-card-title">Before we begin</div>
        <div class="name-card-sub">What should I call you?</div>
    </div>
    """, unsafe_allow_html=True)

    name = st.text_input("", placeholder="Enter your first name…", label_visibility="collapsed")

    if name:
        st.session_state.name = name.strip()
        st.rerun()


# ------------------------------
# CHAT INTERFACE
# ------------------------------

if st.session_state.name:

    # Greeting banner (shown once, above chat)
    st.markdown(f"""
    <div class="greeting-banner">
        <div class="greeting-avatar">🩺</div>
        <div class="greeting-text">
            <strong>Hello, {st.session_state.name}.</strong><br>
            I'm ready to help. Ask me any medical question below.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Render chat history ──
    if st.session_state.messages:
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-row user-row">
                    <div class="msg-bubble user-bubble">
                        <div class="msg-label">You</div>
                        {msg["content"]}
                    </div>
                    <div class="msg-avatar user-avatar">👤</div>
                </div>
                """, unsafe_allow_html=True)

            else:
                # Split source line from answer body
                content = msg["content"]
                source_html = ""
                if "\n\nSource(s):" in content:
                    parts = content.split("\n\nSource(s):", 1)
                    body = parts[0]
                    source_html = f'<div class="source-line">📚 Sources: {parts[1].strip()}</div>'
                elif "\n\nSource:" in content:
                    parts = content.split("\n\nSource:", 1)
                    body = parts[0]
                    source_html = f'<div class="source-line">📚 Source: {parts[1].strip()}</div>'
                else:
                    body = content

                # Format newlines
                body_html = body.replace("\n\n", "<br><br>").replace("\n", "<br>")

                st.markdown(f"""
                <div class="msg-row bot-row">
                    <div class="msg-avatar bot-avatar">🩺</div>
                    <div class="msg-bubble bot-bubble">
                        <div class="msg-label">MedAI</div>
                        {body_html}
                        {source_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Input ──
    query = st.chat_input("Ask a medical question…")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Searching medical knowledge base…"):
            answer = ask_medical_question(query)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()