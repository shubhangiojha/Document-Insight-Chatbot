import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from transformers import pipeline

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

alphabet_text = extract_text_from_pdf("C:/Users/Shubhangi/Desktop/internship/goog-10-k-2023.pdf")
tesla_text = extract_text_from_pdf("C:/Users/Shubhangi/Desktop/internship/tsla-20231231-gen.pdf")
uber_text = extract_text_from_pdf("C:/Users/Shubhangi/Desktop/internship/uber-10-k-2023.pdf")
documents = [alphabet_text, tesla_text, uber_text]

model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = np.array([model.encode(doc) for doc in documents])
doc_embeddings = np.vstack([emb.reshape(1, -1) for emb in doc_embeddings]) 
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)
query = "What does Google do?"
query_vector = model.encode(query).reshape(1, -1) 
distances, indices = index.search(query_vector, k=2)

def search_query(query, model, index, documents, top_k=3):
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    results = [documents[i] for i in indices[0]]
    return results

query = "What are the risk factors associated with Google and Tesla?"
results = search_query(query, model, index, documents)

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def get_llm_response(query, context):
    return qa_pipeline(question=query, context=context)

combined_documents = " ".join([doc[:1000] for doc in [alphabet_text, tesla_text, uber_text]])

query = "What are the risk factors associated with Google and Tesla??"
response = get_llm_response(query, combined_documents)
print("Answer:", response["answer"])