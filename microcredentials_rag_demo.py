# microcredentials_rag_demo.py
import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import tempfile

# Set your OpenAI API key (or use environment variable)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

SUPPORTED_TYPES = {"pdf": PyPDFLoader, "docx": Docx2txtLoader, "txt": TextLoader}

@st.cache_resource
def load_docs(file_path, file_ext):
    loader_class = SUPPORTED_TYPES[file_ext]
    loader = loader_class(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(documents)

@st.cache_resource
def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

st.sidebar.title("📄 Upload Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload PDF, DOCX or TXT documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

st.title("💡 Micro-Credentials Knowledge Assistant")
st.write("Ask anything about micro-credentials and learning pathways in the cultural and creative industries (CCIs).")

if uploaded_files:
    all_chunks = []

    for uploaded_file in uploaded_files:
        suffix = uploaded_file.name.split(".")[-1].lower()
        if suffix not in SUPPORTED_TYPES:
            st.error(f"Unsupported file type: {uploaded_file.name}")
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        chunks = load_docs(tmp_file_path, suffix)
        all_chunks.extend(chunks)

    if not all_chunks:
        st.warning("No valid documents loaded.")
    else:
        vectorstore = build_vectorstore(all_chunks)
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(temperature=0)
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)

        query = st.text_input("What do you want to know?")
        if query:
            result = qa_chain({"question": query})
            st.markdown("**Answer:**")
            st.write(result["answer"])

            if result.get("sources"):
                st.markdown("**Sources:**")
                st.write(result["sources"])
else:
    st.info("👈 Upload one or more documents to begin.")
