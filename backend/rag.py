import faiss
import numpy as np
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# Load RAG model
rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-base")

# Set up retrieval with a simple FAISS index
texts = ["Educational text 1", "Educational text 2", "Educational text 3"]  # Replace with your educational data

# Tokenize the documents
inputs = rag_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
embeddings = rag_model.question_encoder(**inputs).pooler_output.detach().numpy()

# Initialize FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def retrieve_similar_results(query, k=2):
    query_inputs = rag_tokenizer([query], return_tensors="pt")
    query_embedding = rag_model.question_encoder(**query_inputs).pooler_output.detach().numpy()

    distances, indices = index.search(query_embedding, k)  # Retrieve top k results
    return [texts[i] for i in indices[0]]

# Example usage
query = "What is photosynthesis?"
print(retrieve_similar_results(query))