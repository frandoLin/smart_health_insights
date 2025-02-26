# backend/data-ingestion/index_documents.py
# Description: This script is used to generate embeddings for a list of documents using the SentenceTransformer model.

from load_data import load_documents
import faiss
from sentence_transformers import SentenceTransformer
import pickle

def main():
    # Load and split documents from the text files.
    documents = load_documents()
    if not documents:
        raise ValueError("No documents found. Please check your dataset files.")
    
    print(f"Total documents loaded: {len(documents)}")
    
    # Initialize the SentenceTransformer model.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings for each document.
    print("Generating embeddings for documents...")
    embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
    
    # Create a FAISS index using the embeddings.
    dim = embeddings.shape[1]
    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    # Save the FAISS index and the list of documents.
    print("Saving FAISS index and document metadata...")
    faiss.write_index(index, "faiss_index.idx")
    with open("documents.pkl", "wb") as f:
        pickle.dump(documents, f)
    
    print(f"Successfully indexed {len(documents)} documents!")

if __name__ == "__main__":
    main()
