import os
from pypdf import PdfReader
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
api_key =st.secrets["GOOGLE_API_KEY"] 

#yahn pr pdf ka exact name likhna hai
pdf_path = "case .pdf"

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
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
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


