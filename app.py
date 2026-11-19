import streamlit as st
import io
from pypdf import PdfReader
from gtts import gTTS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Page Configuration
st.set_page_config(page_title="Legal AI Case Assistant", page_icon="⚖", layout="wide")
st.title("⚖ AI-Powered Multi-Lingual Legal Case Assistant")
st.markdown("Court case PDF upload karein aur Summary, Timeline, Legal Sections, aur Urdu/English QA hasil karein.")

# Sidebar Settings
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google Gemini API Key:", type="password")
    st.divider()
    language = st.selectbox(
        "Jawab ki Zaban (Language):",
        ["Auto-Detect (Match Question)", "English", "Urdu(اردو)", "Roman Urdu"]
    )
    enable_voice = st.checkbox("Enable AI Voice Output (Audio)", value=True)
    st.divider()
    uploaded_file = st.file_uploader("Court Case PDF Upload Karein", type=["pdf"])
    process_btn = st.button("Analyze Documents", use_container_width=True)

# Helper Functions
def get_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text, len(reader.pages)

def create_vector_store(text, api_key):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )
    return FAISS.from_texts(chunks, embedding=embeddings)

def generate_audio(text, lang_choice):
    tts_lang = 'ur' if "Urdu(اردو)" in lang_choice else 'en'
    clean_text = text.replace("*", "").replace("#", "")[:600]
    tts = gTTS(text=clean_text, lang=tts_lang, slow=False)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

def get_rag_chain(vector_store, api_key, selected_lang):
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0.3
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    template = """
    You are an expert AI Legal Assistant analyzing a court law case.
    Answer strictly based on the provided context.
    Language Instruction: {selected_lang}. (If Auto-Detect, reply in the exact language of the user's question: English, Urdu, or Roman Urdu)
    Simplify complex legal terms (legalese) so anyone can understand.
    Highlight key dates, laws, and sections clearly.

    Context: {context}
    Question: {input}
    Legal Response:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
            "selected_lang": lambda x: selected_lang,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

# Process Button Logic (Fixed single block)
if process_btn:
    if not api_key:
        st.sidebar.error("Pehly Gemini API Key darj karein!")
    elif not uploaded_file:
        st.sidebar.error("Pehly PDF file upload karein!")
    else:
        with st.spinner("Court Document analyze ho raha hai..."):
            try:
                raw_text, page_count = get_pdf_text(uploaded_file)
                if not raw_text.strip():
                    st.sidebar.error("Is PDF mein readable text nahi mila (shayed yeh scanned image hai).")
                else:
                    st.session_state.vector_store = create_vector_store(raw_text, api_key)
                    st.session_state.raw_text = raw_text
                    st.session_state.doc_stats = {
                        "pages": page_count,
                        "words": len(raw_text.split())
                    }
                    st.session_state.messages = []
                    st.sidebar.success(f"Done! {page_count} pages analyzed.")
            except Exception as e:
                st.sidebar.error(f"Error aaya hai: {e}")

# Main App Interface
if "vector_store" in st.session_state:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pages", st.session_state.doc_stats["pages"])
    col2.metric("Total Words", st.session_state.doc_stats["words"])
    col3.metric("Active Mode", language.split()[0])

    if not api_key:
        st.warning("Chat aur summary use karne ke liye sidebar mein API key likhi rehne dein.")
    else:
        rag_chain = get_rag_chain(st.session_state.vector_store, api_key, language)

        tab1, tab2, tab3 = st.tabs(["Legal QA Chat", "Case Summary & Judgment", "Timeline & Legal Sections"])

        with tab1:
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_q = st.chat_input("Case ke baray mein koi bhi sawal poochein (English, Urdu, ya Roman Urdu)...")
            if user_q:
                st.session_state.messages.append({"role": "user", "content": user_q})
                with st.chat_message("user"):
                    st.markdown(user_q)

                with st.chat_message("assistant"):
                    with st.spinner("Jawab tayyar ho raha hai..."):
                        ans = rag_chain.invoke(user_q)
                        st.markdown(ans)
                        if enable_voice:
                            st.audio(generate_audio(ans, language), format="audio/mp3")
                st.session_state.messages.append({"role": "assistant", "content": ans})

        with tab2:
            st.subheader("One-Click Case Breakdown")
            if st.button("Generate Complete Case Summary"):
                with st.spinner("Summary ban rahi hai..."):
                    summary_prompt = (
                        "Provide a structured summary of this court case including: "
                        "1. Case Title & Parties Involved, 2. Core Dispute/Background, "
                        "3. Key Arguments, and 4. Final Court Judgment/Decision."
                    )
                    summary_res = rag_chain.invoke(summary_prompt)
                    st.markdown(summary_res)
                    if enable_voice:
                        st.audio(generate_audio(summary_res, language), format="audio/mp3")

        with tab3:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Chronological Case Timeline")
                if st.button("Extract All Dates & Events"):
                    with st.spinner("Timeline nikal rahe hain..."):
                        time_res = rag_chain.invoke(
                            "List all important dates and events in chronological order from this document."
                        )
                        st.markdown(time_res)

            with col_b:
                st.subheader("Law, Acts & Sections Cited")
                if st.button("Extract Legal Citations"):
                    with st.spinner("Laws aur sections dhoond rahe hain..."):
                        law_res = rag_chain.invoke(
                            "List all laws, acts, articles, or legal sections (e.g., PPC, CrPC, CPC, Constitution) "
                            "cited in this case and briefly explain what each means."
                        )
                        st.markdown(law_res)
else:
    st.info("Shuru karne ke liye left sidebar mein apni **Gemini API Key** dalein aur **Court Case PDF** upload karein!")






