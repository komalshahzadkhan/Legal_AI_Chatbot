import os
from pypdf import PdfReader
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
api_key =st.secrets["GOOGLE_API_KEY"] 

#yahn pr pdf ka exact name likhna hai
pdf_path = "case.pdf"

#yahn hum pdf se text nikaly gy
reader = PdfReader(pdf_path)
raw_text= ""
for page in reader.pages:
  raw_text +=page.extract_text() or ""

#text ko choty hisson main tory gy yani chunk kary gy
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs =text_splitter.create_documents([raw_text])

#vector database (Memory) banana
raw_text =""
for page in reader.pages:
  raw_text +=page.extract_text()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.create_documents([raw_text])
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = Chroma.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k":4})



#gemini model aur vector store (memory) Banana
llm =ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

system_prompt =(
    "You are an expert legal assistant. Use the following pieces of retrieved context to answer"
    "the question. Answer the user in the language they ask (English, Urdu, or Roman Urdu)"
    "If the question is in Urdu or Roman Urdu, translate the complex legal terms into simple language so anyone can understand.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
print("Setup mukammal ho gaya hai! koi error nhi aya")

st.set_page_config(page_title="Legal AI Chatbot", page_icon="⚖️")
st.title("⚖️ Legal AI Chatbot")

# 1. Language Selector (English, Urdu, Roman Urdu)
language = st.selectbox(
    "🌐 Select Response Language:", ["English", "Urdu (اردو)", "Roman Urdu"]
)

# 2. One-Click Action Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("📄 Summarize Case"):
        with st.spinner("Summarizing case..."):
            query = f"Provide a complete, clear summary of this court case in {language} language."
            res = retrieval_chain.invoke({"input": query})
            st.markdown("### 📄 Case Summary")
            st.write(res["answer"])

with col2:
    if st.button("⭐ Extract Important Info"):
        with st.spinner("Extracting key details..."):
            query = f"Extract all important information (Parties involved, Dates, Case Number, Sections/Laws, and Final Verdict) from this case in {language} language using bullet points."
            res = retrieval_chain.invoke({"input": query})
            st.markdown("### ⭐ Key Case Information")
            st.write(res["answer"])

st.divider()

# 3. Custom Q&A in Selected Language
user_question = st.text_input(
    f"💬 Ask any question about the case (Reply will be in {language}):"
)

if user_question:
    with st.spinner("Searching case records..."):
        full_query = f"Answer this question strictly based on the case document in {language} language: {user_question}"
        response = retrieval_chain.invoke({"input": full_query})
        st.write(response["answer"])

