import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PyPDF2 import PdfReader

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def build_faiss_index(documents, _model):
    doc_embeddings = np.array([_model.encode(doc) for doc in documents])
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    return index

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def search_query(query, model, index, documents, top_k=3):
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    results = [documents[i] for i in indices[0]]
    return " ".join([res[:2000] for res in results])  

def get_llm_response(query, documents, model, index):
    context = search_query(query, model, index, documents)
    response = qa_pipeline(question=query, context=context)
    return response["answer"]

st.title("Document Insight Chatbot")
st.sidebar.header("Document Uploader")
uploaded_files = st.sidebar.file_uploader("Upload PDF files (Max: 3)", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    documents = [extract_text_from_pdf(file) for file in uploaded_files]
    model = load_model()
    index = build_faiss_index(documents, model)
    st.sidebar.write("Files loaded successfully!")
    
    query = st.text_input("Enter your query:", "")
    if query:
        response = get_llm_response(query, documents, model, index)
        st.subheader("Answer:")
        st.write(response)
else:
    st.warning("Please upload documents to start.")
