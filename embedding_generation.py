import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return np.array(embeddings)

def store_in_faiss(embeddings):
    """
    Store embeddings in a FAISS index for efficient similarity search.

    Args:
        embeddings (np.array): Numpy array of embeddings.

    Returns:
        faiss.IndexFlatL2: FAISS index with stored embeddings.
    """
    d = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    print("Embeddings added to FAISS index.")
    return index

if __name__ == "__main__":
    
    chunks = [
        "The unemployment rate for Bachelor's degree holders is 4%.",
        "Master's degree holders have a 3.5% unemployment rate.",
        "Doctoral degree holders have the lowest rate at 2.1%."
    ]

    # Step 1: Generate embeddings
    embeddings = generate_embeddings(chunks)
    print(f"Generated Embeddings Shape: {embeddings.shape}")

    # Step 2: Store embeddings in FAISS
    index = store_in_faiss(embeddings)

    # Example Query
    query = "What is the unemployment rate for Master's degree holders?"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding)

    # Search FAISS index
    k = 1  # Retrieve top 1 result
    distances, indices = index.search(query_embedding, k)
    print(f"Top result: {chunks[indices[0][0]]}")
