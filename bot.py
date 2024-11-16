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

# Example: Extract text from PDF files
alphabet_text = extract_text_from_pdf("C:/Users/Shubhangi/Desktop/internship/goog-10-k-2023.pdf")
tesla_text = extract_text_from_pdf("C:/Users/Shubhangi/Desktop/internship/tsla-20231231-gen.pdf")
uber_text = extract_text_from_pdf("C:/Users/Shubhangi/Desktop/internship/uber-10-k-2023.pdf")
documents = [alphabet_text, tesla_text, uber_text]

model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for the documents
doc_embeddings = np.array([model.encode(doc) for doc in documents])

# Ensure embeddings are 2D (reshape if necessary)
doc_embeddings = np.vstack([emb.reshape(1, -1) for emb in doc_embeddings])  # Stacks all into 2D

# Initialize FAISS vector store
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add vectors to the FAISS index
index.add(doc_embeddings)

# Query FAISS
query = "What does Google do?"
query_vector = model.encode(query).reshape(1, -1)  # Ensure the query is 2D
distances, indices = index.search(query_vector, k=2)

def search_query(query, model, index, documents, top_k=3):
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    results = [documents[i] for i in indices[0]]
    return results

# Example Query
query = "What are the risk factors associated with Google and Tesla?"
results = search_query(query, model, index, documents)

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Function to process a query
def get_llm_response(query, context):
    return qa_pipeline(question=query, context=context)

# Combine documents for context
combined_documents = " ".join([doc[:1000] for doc in [alphabet_text, tesla_text, uber_text]])  # Truncate for efficiency

# Example query
query = "What are the risk factors associated with Google and Tesla??"
response = get_llm_response(query, combined_documents)
print("Answer:", response["answer"])