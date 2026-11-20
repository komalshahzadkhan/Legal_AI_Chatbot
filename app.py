import io
import os
from gtts import gTTS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import streamlit as st

st.set_page_config(page_title="LEGAL AI CASE ASSISTANT", page_icon="⚖️", layout="wide")
st.caption("Upload court case judgments to generate executive summaries, timelines, legal sections, and multilingual Q&A."

# API Key from Streamlit Secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- LEFT SIDEBAR (SETTINGS) ---
with st.sidebar:
    st.header("Settings")
    st.success("API Key Connected (Secure)")

    language = st.selectbox(
        "Jawab ki Zaban (Language):", ["English", "Urdu", "Roman Urdu"]
    )

    enable_audio = st.checkbox("Enable AI Voice Output (Audio)", value=True)

    uploaded_file = st.file_uploader("Court Case PDF Upload Karein", type="pdf")


# Voice Generator Function
def speak_text(text, lang_choice):
    try:
        lang_code = "ur" if lang_choice == "Urdu" else "en"
        tts = gTTS(text=text[:600], lang=lang_code)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        st.audio(audio_fp, format="audio/mp3")
    except Exception:
        pass


# Load & Cache PDF
@st.cache_resource
def process_pdf(file_bytes, file_name):
    if file_bytes is not None:
        reader = PdfReader(io.BytesIO(file_bytes))
    else:
        reader = PdfReader("case.pdf")

    num_pages = len(reader.pages)
    full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
    word_count = len(full_text.split())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    vectorstore = Chroma.from_texts(chunks, embeddings)
    return vectorstore.as_retriever(), num_pages, word_count


file_bytes = uploaded_file.getvalue() if uploaded_file else None
file_name = uploaded_file.name if uploaded_file else "case.pdf"

retriever, total_pages, total_words = process_pdf(file_bytes, file_name)

# Setup LLM & Chain
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

prompt = ChatPromptTemplate.from_template(
    """You are an expert Legal AI Assistant. Answer strictly based on the provided context.
Respond in {language} language.

Context:
{context}

Question: {input}"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- MAIN INTERFACE (EXACTLY LIKE YOUR SCREENSHOT) ---
st.title("Assistant")
st.caption(
    "Court case PDF upload karein aur Summary, Timeline, Legal Sections, aur Urdu/English QA hasil karein."
)

# 3 Metrics Row
m1, m2, m3 = st.columns(3)
m1.metric("Total Pages", total_pages)
m2.metric("Total Words", total_words)
m3.metric("Active Mode", language.split()[0])

st.divider()

# 3 Tabs
tab1, tab2, tab3 = st.tabs([
    "Legal QA Chat",
    "Case Summary & Judgment",
    "Timeline & Legal Sections",
])

# TAB 1: Legal QA Chat
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input(f"Apna sawal poochein ({language} mein)...")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Jawab tayyar ho raha hai..."):
                res = retrieval_chain.invoke(
                    {"input": user_q, "language": language}
                )
                ans = res["answer"]
                st.markdown(ans)
                if enable_audio:
                    speak_text(ans, language)
        st.session_state.messages.append({"role": "assistant", "content": ans})

# TAB 2: Case Summary & Judgment
with tab2:
    st.subheader("One-Click Case Breakdown")
    if st.button("Generate Complete Case Summary"):
        with st.spinner("Case ka mukammal khulasa ban raha hai..."):
            q = (
                "Provide a structured Case Breakdown with numbered headings: "
                "1. Case Title & Parties Involved, 2. Background & Key Facts, "
                "3. Legal Issues, 4. Final Judgment & Verdict."
            )
            res = retrieval_chain.invoke({"input": q, "language": language})
            st.markdown(res["answer"])
            if enable_audio:
                speak_text(res["answer"], language)

# TAB 3: Timeline & Legal Sections
with tab3:
    st.subheader("Important Dates, Timeline & Legal Sections")
    if st.button("Extract Timeline & Legal Sections"):
        with st.spinner("Important information nikali ja rahi hai..."):
            q = (
                "Extract a chronological Timeline of Important Dates and list all "
                "Legal Sections, Acts, and Citations mentioned in this document."
            )
            res = retrieval_chain.invoke({"input": q, "language": language})
            st.markdown(res["answer"])
            if enable_audio:
                speak_text(res["answer"], language)
