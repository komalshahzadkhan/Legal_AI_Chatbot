import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from gtts import gTTS
import io

st.set_page_config(
    page_title="LEGAL AI | Case Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .top-badge {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 3px;
        color: #D4AF37;
        text-transform: uppercase;
        margin-bottom: -12px;
        line-height: 1.2;
    }
    .main-heading {
        font-size: 3.3rem;
        font-weight: 900;
        letter-spacing: 1.5px;
        color: #FFFFFF;
        text-transform: uppercase;
        margin-top: 0px;
        margin-bottom: 8px;
        line-height: 1.1;
    }
    .sub-heading {
        font-size: 1.08rem;
        color: #A0AEC0;
        margin-bottom: 25px;
    }
    .welcome-card {
        background-color: #161922;
        padding: 50px 30px;
        border-radius: 14px;
        border: 1px solid #2D3748;
        text-align: center;
        margin-top: 25px;
    }
    .feature-pill {
        display: inline-block;
        background-color: #222736;
        color: #E2E8F0;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        margin: 6px;
        border: 1px solid #374151;
    }
    </style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text, len(reader.pages)

def generate_audio(text, lang_choice):
    try:
        tts_lang = "ur" if lang_choice == "Urdu" else "en"
        clean_text = text.replace("*", "").replace("#", "")[:1000]
        tts = gTTS(text=clean_text, lang=tts_lang, slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception:
        return None

# --- AUTOMATIC SECRET KEY CHECK (CLIENT-READY) ---
secret_key = ""
try:
    if "GEMINI_API_KEY" in st.secrets:
        secret_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    secret_key = ""

with st.sidebar:
    st.markdown("## ⚙️ SETTINGS")
    
    if secret_key:
        api_key = secret_key
    else:
        api_key = st.text_input("Google Gemini API Key", type="password", placeholder="Enter your API key...")
        st.divider()
        
    output_language = st.selectbox("Response Language", ["English", "Urdu", "Roman Urdu"], index=0)
    enable_audio = st.checkbox("Enable AI Voice Output (Audio)", value=False)
    st.divider()
    st.markdown("### 📄 DOCUMENT UPLOAD")
    uploaded_file = st.file_uploader("Upload Court Case Document (PDF)", type=["pdf"])

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

# --- LARGE CAPITAL TWO-LINE HEADER ---
st.markdown('<p class="top-badge">⚖️ LEGAL AI</p>', unsafe_allow_html=True)
st.markdown('<h1 class="main-heading">CASE ASSISTANT</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-heading">Upload court case PDFs to generate executive summaries, case timelines, legal sections, and multilingual client Q&A.</p>', unsafe_allow_html=True)

if uploaded_file is not None:
    if not api_key:
        st.warning("⚠️ Please configure your API Key in Streamlit Secrets or enter it in the sidebar.")
    else:
        if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
            with st.spinner("Processing legal document..."):
                text, num_pages = extract_text_from_pdf(uploaded_file)
                st.session_state.pdf_text = text
                st.session_state.num_pages = num_pages
                st.session_state.word_count = len(text.split())
                st.session_state.current_file = uploaded_file.name
                st.session_state.chat_history = []
                st.session_state.summary_output = None
                st.session_state.timeline_output = None

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Pages", value=st.session_state.num_pages)
        with col2:
            st.metric(label="Total Words", value=f"{st.session_state.word_count:,}")
        with col3:
            st.metric(label="Active Mode", value=output_language)

        st.divider()

        tab1, tab2, tab3 = st.tabs([
            "💬 Legal Q&A Chat",
            "📑 Case Summary & Judgment",
            "⏳ Timeline & Legal Sections"
        ])

        with tab1:
            st.subheader("Client Legal Q&A Assistant")
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            user_question = st.chat_input("Ask any question about this legal document...")
            if user_question:
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing document and drafting response..."):
                        prompt = f"You are a senior Legal AI Consultant helping a client understand a legal document. Answer strictly based on the text in {output_language}.\n\nDocument Text:\n{st.session_state.pdf_text[:28000]}\n\nClient Question: {user_question}"
                        response = model.generate_content(prompt)
                        reply = response.text
                        st.markdown(reply)
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                        if enable_audio:
                            audio_bytes = generate_audio(reply, output_language)
                            if audio_bytes:
                                st.audio(audio_bytes, format="audio/mp3")

        with tab2:
            st.subheader("One-Click Case Breakdown")
            if st.button("Generate Complete Case Summary", type="primary"):
                with st.spinner("Generating comprehensive client breakdown..."):
                    summary_prompt = f"Analyze the legal document and create a client-friendly breakdown in {output_language} with these headings:\n1. Case Title & Parties Involved\n2. Executive Summary (Plain & Easy Explanation)\n3. Key Facts & Background\n4. Core Legal Issues\n5. Arguments Presented\n6. Final Judgment / Court Verdict\n7. Key Takeaways for the Client\n\nDocument Text:\n{st.session_state.pdf_text[:28000]}"
                    response = model.generate_content(summary_prompt)
                    st.session_state.summary_output = response.text

            if st.session_state.summary_output:
                st.markdown(st.session_state.summary_output)
                if enable_audio:
                    audio_bytes = generate_audio(st.session_state.summary_output, output_language)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

        with tab3:
            st.subheader("Case Chronology & Applicable Laws")
            if st.button("Extract Timeline & Legal Sections", type="primary"):
                with st.spinner("Extracting dates, events, and legal provisions..."):
                    timeline_prompt = f"From the legal document, extract in {output_language}:\n1. Chronological Timeline of Events\n2. Cited Legal Sections, Acts & Statutes (with simple explanation)\n3. Key Precedents Referred\n\nDocument Text:\n{st.session_state.pdf_text[:28000]}"
                    response = model.generate_content(timeline_prompt)
                    st.session_state.timeline_output = response.text

            if st.session_state.timeline_output:
                st.markdown(st.session_state.timeline_output)

else:
    st.markdown('<div class="welcome-card"><h2 style="color: #FFFFFF; margin-bottom: 10px;">👈 Upload a Legal Document to Get Started</h2><p style="color: #A0AEC0; font-size: 1.05rem; max-width: 650px; margin: 0 auto 25px auto;">Select any court judgment, legal notice, or case file (PDF) from the left sidebar to unlock instant AI-powered legal insights for your clients.</p><div><span class="feature-pill">💬 Interactive Legal Q&A</span><span class="feature-pill">📑 One-Click Case Breakdown</span><span class="feature-pill">⏳ Event Timeline & Legal Sections</span><span class="feature-pill">🌐 English, Urdu & Roman Urdu</span></div></div>', unsafe_allow_html=True)
