import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from gtts import gTTS
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="LEXIS AI | Legal Case Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM HIGH-CLASS STYLING ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #FFFFFF;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #A0AEC0;
        margin-bottom: 25px;
    }
    .welcome-box {
        background-color: #1E212B;
        padding: 45px;
        border-radius: 12px;
        border: 1px solid #2D3748;
        text-align: center;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SECURE BACKEND API CONFIGURATION (HIDDEN FROM UI) ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception:
    st.error("⚠️ Configuration Error: Please add 'GEMINI_API_KEY' in Streamlit Secrets.")
    st.stop()

# --- HELPER FUNCTIONS ---
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

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("## ⚙️ CONFIGURATION")
    st.divider()
    
    output_language = st.selectbox(
        "Select Output Language",
        ["English", "Urdu", "Roman Urdu"],
        index=0
    )
    
    enable_audio = st.checkbox("Enable AI Voice Output (Audio)", value=False)
    
    st.divider()
    st.markdown("### 📄 DOCUMENT UPLOAD")
    uploaded_file = st.file_uploader(
        "Upload Court Case Judgment (PDF)",
        type=["pdf"],
        help="Upload a court case PDF to unlock AI analysis tools."
    )

# --- MAIN HEADER ---
st.markdown('<p class="main-title">⚖️ LEGAL AI CASE ASSISTANT</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload court case judgments to generate executive summaries, timelines, legal sections, and multilingual Q&A.</p>', unsafe_allow_html=True)

# --- CONDITIONAL DISPLAY: ONLY SHOW TOOLS AFTER PDF UPLOAD ---
if uploaded_file is not None:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Analyzing legal document..."):
            text, num_pages = extract_text_from_pdf(uploaded_file)
            st.session_state.pdf_text = text
            st.session_state.num_pages = num_pages
            st.session_state.word_count = len(text.split())
            st.session_state.current_file = uploaded_file.name
            st.session_state.chat_history = []
            st.session_state.summary_output = None
            st.session_state.timeline_output = None

    # --- DOCUMENT METRICS BAR ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Pages", value=st.session_state.num_pages)
    with col2:
        st.metric(label="Total Words", value=f"{st.session_state.word_count:,}")
    with col3:
        st.metric(label="Active Output Mode", value=output_language)

    st.divider()

    # --- NAVIGATION TABS ---
    tab1, tab2, tab3 = st.tabs([
        "💬 Legal Q&A Chat",
        "📑 Case Summary & Judgment",
        "⏳ Timeline & Legal Sections"
    ])

    # TAB 1: LEGAL Q&A CHAT
    with tab1:
        st.subheader("Interactive Case Q&A")
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input("Ask a legal question about this case...")
        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Drafting legal response..."):
                    prompt = f"""
                    You are an expert Legal AI Assistant. Answer the user's question strictly based on the provided court case text.
                    Provide the response in {output_language}.
                    
                    Court Case Text:
                    {st.session_state.pdf_text[:25000]}
                    
                    User Question: {user_question}
                    """
                    response = model.generate_content(prompt)
                    reply = response.text
                    st.markdown(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    
                    if enable_audio:
                        audio_bytes = generate_audio(reply, output_language)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")

    # TAB 2: ONE-CLICK CASE SUMMARY & JUDGMENT
    with tab2:
        st.subheader("One-Click Case Breakdown")
        if st.button("Generate Complete Case Summary", type="primary"):
            with st.spinner("Extracting case details and verdict..."):
                summary_prompt = f"""
                Analyze the following legal document and provide a structured breakdown in {output_language} with the following clear headings:
                1. Case Title & Parties Involved
                2. Background & Key Facts
                3. Core Legal Issues
                4. Arguments Presented
                5. Final Judgment / Verdict
                
                Document Text:
                {st.session_state.pdf_text[:25000]}
                """
                response = model.generate_content(summary_prompt)
                st.session_state.summary_output = response.text

        if st.session_state.summary_output:
            st.markdown(st.session_state.summary_output)
            if enable_audio:
                audio_bytes = generate_audio(st.session_state.summary_output, output_language)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")

    # TAB 3: TIMELINE & LEGAL SECTIONS
    with tab3:
        st.subheader("Chronological Timeline & Cited Laws")
        if st.button("Extract Timeline & Legal Sections", type="primary"):
            with st.spinner("Identifying dates, events, and statutes..."):
                timeline_prompt = f"""
                From the provided legal document, extract the following in {output_language}:
                1. Chronological Timeline of Events (Dates and what happened)
                2. Cited Legal Sections, Acts, and Precedents
                
                Document Text:
                {st.session_state.pdf_text[:25000]}
                """
                response = model.generate_content(timeline_prompt)
                st.session_state.timeline_output = response.text

        if st.session_state.timeline_output:
            st.markdown(st.session_state.timeline_output)

else:
    # --- CLEAN WELCOME SCREEN WHEN NO PDF IS UPLOADED ---
    st.markdown("""
        <div class="welcome-box">
            <h3>👈 Upload a Court Case PDF from the Sidebar to Begin</h3>
            <p style="color: #A0AEC0; margin-top: 10px;">
                Once uploaded, you will unlock instant <b>Case Summaries</b>, <b>Legal Sections</b>, 
                <b>Event Timelines</b>, and <b>Multilingual Q&A</b> in English, Urdu, and Roman Urdu.
            </p>
        </div>
    """, unsafe_allow_html=True)
