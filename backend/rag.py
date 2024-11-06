#!pip install faiss-cpu transformers

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

# Query example
query = "What is photosynthesis?"
query_inputs = rag_tokenizer([query], return_tensors="pt")
query_embedding = rag_model.question_encoder(**query_inputs).pooler_output.detach().numpy()

# Retrieve and print results
distances, indices = index.search(query_embedding, k=2)  # Find top 2 similar results
print([texts[i] for i in indices[0]])
